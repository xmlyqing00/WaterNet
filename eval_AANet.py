import os
import argparse
import sys
import time
import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage

from src.AANet.feature_net import FeatureNet
from src.AANet.features import FeatureTemplates, split_mask, split_features, calc_similarity
from src.AANet.dataset import WaterDataset
from src.avg_meter import AverageMeter
from src.cvt_images_to_overlays import run_cvt_images_to_overlays
from src.utils import load_image_in_PIL, iou_tensor

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


def eval_AANetNet():
    
    torch.set_printoptions(precision=3, threshold=3000, linewidth=160, sci_mode=False)
    np.set_printoptions(precision=3, threshold=3000, linewidth=160, suppress=False)

    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    if sys.platform == 'darwin':
        cfg_dataset = 'dataset_mac'
    elif sys.platform == 'linux':
        cfg_dataset = 'dataset_ubuntu'

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch AANet Testing')
    parser.add_argument(
        '-b', '--benchmark', action='store_true',
        help='Evaluate the video with groundtruth.')
    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    parser.add_argument(
        '-v', '--video-name', default=None, type=str,
        help='Test video name (default: none).')
    parser.add_argument(
        '-m', '--model-name', default='AANet', type=str,
        help='Model name for the ouput segmentation, it will create a subfolder under the out_folder.')
    parser.add_argument(
        '-o', '--out-folder', default=cfg['paths'][cfg_dataset], type=str, metavar='PATH',
        help='Folder for the output segmentations.')
    args = parser.parse_args()

    print('Args:', args)

    if args.checkpoint is None:
        raise ValueError('Must input checkpoint path.')
    if args.video_name is None:
        raise ValueError('Must input video name.')

    water_thres = 0.5

    # Dataset
    dataset_args = {}
    if torch.cuda.is_available():
        dataset_args = {
            'num_workers': 1, # int(cfg['params_AA']['num_workers']),
            'pin_memory': False # bool(cfg['params_AA']['pin_memory'])
        }

    dataset = WaterDataset(
        mode='eval',
        dataset_path=cfg['paths']['dataset_ubuntu'],
        input_size=(int(cfg['params_AA']['input_w']), int(cfg['params_AA']['input_h'])),
        test_case=args.video_name
    )
    
    eval_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        **dataset_args
    )

    # Load pretrained model
    # Set ouput path
    setting_prefix = args.model_name

    feature_templates = FeatureTemplates(args.checkpoint)
    f_obj, f_bg = feature_templates.build_from_eval(args.video)
    
    out_path = os.path.join(args.out_folder, setting_prefix + '_segs', args.video_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Start testing
    feature_net.to(device).eval()
    deconv_net.to(device).eval()
    running_time = AverageMeter()
    running_endtime = time.time()
    
    # Get the first frame annotation
    first_frame_mask = dataset.get_first_frame_label()
    first_frame_seg = TF.to_pil_image(first_frame_mask)
    first_frame_seg.save(os.path.join(out_path, '0.png'))
    first_frame_mask = first_frame_mask.to(device).unsqueeze(0)

    # Get the first frame features
    first_frame = dataset.get_first_frame().to(device).unsqueeze(0)
    eval_size = first_frame.shape[-2:]
    f3, f0, f1, f2 = feature_net(first_frame)
    # f0 = F.interpolate(f0, size=f0.shape[-2:], mode='bilinear')
    # f1 = F.interpolate(f1, size=f0.shape[-2:], mode='bilinear')
    f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
    f3 = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=False)
    feature0 = torch.cat((f1, f2, f3), 1)
    # print(f0.shape)
    # print(f1.shape)
    # print(f2.shape)
    # print(f3.shape)
    # print(feature0.shape)
    # Normalize features. Size: (c, h, w)
    feature_map = feature0.detach().squeeze(0)
    feature_norms = feature_map.norm(p=2, dim=0, keepdim=True)
    feature_map = feature_map / feature_norms
    c, h, w = feature_map.shape
    feature_n = h * w

    print('c', c, 'h', h, 'w', w)

    # Get template fetures. Size: (c, m0), (c, m1)
    pre_frame_mask = F.interpolate(first_frame_mask, size=(h, w), mode='bilinear', align_corners=False).detach()
    # pre_frame_mask Size (1, 1, h, w)
    feature_templates_obj, feature_templates_bg = split_features(feature_map, pre_frame_mask)
    m0 = feature_templates_obj.shape[1]
    m1 = feature_templates_bg.shape[1]

    m0_first, m1_first = m0, m1

    templates_n = [(m0, m1)]

    print('Init features', feature_templates_obj.shape, feature_templates_bg.shape)

    keep_features_n = 7

    avg_iou = 0

    if args.benchmark:
        gt_folder = os.path.join(cfg['paths'][cfg_dataset], 'test_annots', args.video_name)
        gt_list = os.listdir(gt_folder)
        gt_list.sort(key = lambda x: (len(x), x))
        gt_list.pop(0)

    print('Erosion params:', int(cfg['params_AA']['r0']), int(cfg['params_AA']['r1']))
    with torch.no_grad():
        for i, sample in enumerate(eval_loader):

            img = sample['img'].to(device)     

            f3, f0, f1, f2 = feature_net(img)
            # f0 = F.interpolate(f0, size=f0.shape[-2:], mode='bilinear')
            # f1 = F.interpolate(f1, size=f0.shape[-2:], mode='bilinear')
            f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            f3 = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            feature_map = torch.cat((f1, f2, f3), 1)
            # output = deconv_net(f3, f0, f1, f2, img.shape).detach() # Size: (1, 1, h, w)

            feature_map = feature_map.detach().squeeze(0)
            feature_map /= feature_map.norm(p=2, dim=0, keepdim=True) # Size: (c, h, w)

            # if not args.no_conf and not args.no_aa:
            #     # Add center features to template features
            #     center_features_obj, center_features_bg = split_features(feature_map, pre_frame_mask, erosion_iters=int(cfg['params_AA']['r0']))
            #     feature_templates_obj = torch.cat((feature_templates_obj, center_features_obj), dim=1)
            #     feature_templates_bg = torch.cat((feature_templates_bg, center_features_bg), dim=1)
            #     print('Conf features:\t', center_features_obj.shape, center_features_bg.shape)
            #     print('Added conf features:\t', feature_templates_obj.shape, feature_templates_bg.shape)

            if not args.no_aa:
                cur_feature = feature_map.reshape((c, feature_n)).transpose(0, 1)
                scores_obj = compute_similarity(cur_feature, feature_templates_obj, (h,w), img.shape[2:])
                scores_bg = compute_similarity(cur_feature, feature_templates_bg, (h,w), img.shape[2:])
                scores = torch.cat((scores_bg, scores_obj), dim=1)
                adaption_seg = F.softmax(scores, dim=1).argmax(dim=1).type(torch.float).detach() # Size: (1, h, w)
                # print(scores_bg)
                # print(scores_obj)

                # adaption_seg = F.softmax(scores, dim=1)[0,1].detach() # Size: (1, 1, h, w)

                final_seg = adaption_seg.unsqueeze(0)
                # print(final_seg.shape)
                # final_seg = (1-l) * output + l * adaption_seg
            # else:
                # final_seg = output

            # final_seg = torch.where(final_seg > 0.5, one_tensor, zero_tensor) # Size: (1, 1, h, w)
            pre_frame_mask = F.interpolate(final_seg, size=(h, w), mode='bilinear', align_corners=False)
            
            # if not args.no_conf and not args.no_aa:
            #     # Remove high-confidence templates
            #     feature_templates_obj = feature_templates_obj[:,:m0]
            #     feature_templates_bg = feature_templates_bg[:,:m1]
            #     print('Removed high-conf features:\t', feature_templates_obj.shape, feature_templates_bg.shape)
            
            # if not args.no_temporal and not args.no_aa:

            #     # Remove previous feature templates
            #     if i > keep_features_n:
            #         j = i - keep_features_n
            #         print(templates_n[j])
            #         print(m1_first)
            #         print(feature_templates_bg.shape)
            #         feature_templates_obj = torch.cat((feature_templates_obj[:,:m0_first], feature_templates_obj[:,m0_first + templates_n[j][0]:]), dim=1)
            #         feature_templates_bg = torch.cat((feature_templates_bg[:,:m1_first], feature_templates_bg[:,m1_first + templates_n[j][1]:]), dim=1)

            #         print('Removed old feature templates.', feature_templates_obj.shape[1], feature_templates_bg.shape[1])

            #     # Add current features to template features
            #     cur_features_obj, cur_features_bg = split_features(feature_map, pre_frame_mask, erosion_iters=int(cfg['params_AA']['r1']))
            #     print('cur features:\t', cur_features_obj.shape, cur_features_bg.shape)
            #     feature_templates_obj = torch.cat((feature_templates_obj, cur_features_obj), dim=1)
            #     feature_templates_bg = torch.cat((feature_templates_bg, cur_features_bg), dim=1)
            #     m0_cur, m1_cur = cur_features_obj.shape[1], cur_features_bg.shape[1]
            #     m0, m1 = feature_templates_obj.shape[1], feature_templates_bg.shape[1]

            #     templates_n.append((m0_cur, m1_cur))
            #     print('template features:\t', feature_templates_obj.shape, feature_templates_bg.shape)


            # print('output', output)
            # print('adapt', adaption_seg)
            # print('final', final_seg)

            if args.benchmark:
                gt_seg = load_image_in_PIL(os.path.join(gt_folder, gt_list[i])).convert('L')
                gt_seg.thumbnail((int(cfg['params_AA']['eval_h']), int(cfg['params_AA']['eval_w'])), Image.ANTIALIAS)
                gt_tf = TF.to_tensor(gt_seg).to(device).type(torch.int)

                iou = iou_tensor(final_seg.squeeze(0).type(torch.int), gt_tf)
                avg_iou += iou.item()
                print('iou:', iou.item())

            if not args.no_aa:

                fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

                seg0 = TF.to_pil_image(scores_obj.squeeze(0).cpu())
                axes[0].imshow(seg0, cmap='plasma', interpolation='nearest', vmin=0, vmax=255)
                seg0.save(os.path.join(out_path, 'obj_%d.png' % (i + 1)))

                seg1 = TF.to_pil_image(scores_bg.squeeze(0).cpu())
                axes[1].imshow(seg1, cmap='plasma', interpolation='nearest', vmin=0, vmax=255)
                seg1.save(os.path.join(out_path, 'bg_%d.png' % (i + 1)))

                final_seg = TF.to_pil_image(final_seg.squeeze(0).cpu())
                axes[2].imshow(final_seg)
                final_seg.save(os.path.join(out_path, '%d.png' % (i + 1)))
                plt.savefig(os.path.join(out_path, 'heatmap_%d.png' %(i + 1)))
                # plt.show()
                plt.close(fig)
            else:
                final_seg = TF.to_pil_image(final_seg.squeeze(0).cpu())
                final_seg.save(os.path.join(out_path, '%d.png' % (i + 1)))

            running_time.update(time.time() - running_endtime)
            running_endtime = time.time()


            print('Segment: [{0:4}/{1:4}]\t'
                'Time: {running_time.val:.3f}s ({running_time.sum:.3f}s)\t'.format(
                i + 1, len(eval_loader), running_time=running_time))


    if args.benchmark:
        print('total_iou:', avg_iou)
        avg_iou /= len(eval_loader)
        print('avg_iou:', avg_iou, 'frame_num:', len(eval_loader))
        
        # scores_path = os.path.join(args.out_folder, 'scores.csv')
        # print(scores_path)
        # if os.path.exists(scores_path):
        #     scores_df = pd.read_csv(scores_path)
        #     scores_df.set_index(['AANet', 'AANet_no_conf'])
        # else:
        #     scores_df = pd.DataFrame({'a':None, 'b':None})
        # print(scores_df)
        # scores_df[setting_prefix][args.video_name] = avg_iou
        # print(scores_df)
        # scores_df[setting_prefix + '_total'][args.video_name] = len(eval_loader)
        # scores_df.to_csv(scores_path)

    run_cvt_images_to_overlays(args.video_name, args.out_folder, setting_prefix, eval_size)
    
if __name__ == '__main__':
    eval_AANetNet()

