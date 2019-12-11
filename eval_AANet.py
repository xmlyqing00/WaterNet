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
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage

from src.AANet import FeatureNet, DeconvNet
from src.dataset import WaterDataset_RGB
from src.avg_meter import AverageMeter
from src.cvt_images_to_overlays import run_cvt_images_to_overlays
from src.utils import load_image_in_PIL, iou_tensor

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

one_tensor = torch.ones(1).to(device)
zero_tensor = torch.zeros(1).to(device)

# Hyper parameters
erosion_bg_factor = 1.3

def split_mask(mask, split_thres, erosion_iters):

    obj_mask = torch.where(mask > split_thres, one_tensor, zero_tensor)
    bg_mask = torch.where(mask < (1-split_thres), one_tensor, zero_tensor)

    if erosion_iters > 0:
        obj_mask = obj_mask.squeeze().cpu().numpy()
        bg_mask = bg_mask.squeeze().cpu().numpy()
        
        erosion_obj = erosion_iters
        erosion_bg = int(erosion_iters * erosion_bg_factor)

        # print('erosion obj', erosion_obj, 'erosion bg', erosion_bg)

        obj_mask_pad = np.pad(obj_mask, ((erosion_obj, erosion_obj), (erosion_obj, erosion_obj)), 'edge')
        bg_mask_pad = np.pad(bg_mask, ((erosion_bg, erosion_bg), (erosion_bg, erosion_bg)), 'edge')

        # print('obj_pad', obj_mask_pad)
        # print('bg_pad', bg_mask_pad)

        center_mask_obj = ndimage.binary_erosion(obj_mask_pad, iterations=erosion_obj).astype(np.float32)
        center_mask_bg = ndimage.binary_erosion(bg_mask_pad, iterations=erosion_bg).astype(np.float32)
        
        # print('obj_erosion', center_mask_obj)
        # print('bg_erosion', center_mask_bg)

        center_mask_obj = center_mask_obj[erosion_obj: erosion_obj + obj_mask.shape[0], erosion_obj: erosion_obj + obj_mask.shape[1]]
        center_mask_bg = center_mask_bg[erosion_bg: erosion_bg + bg_mask.shape[0], erosion_bg: erosion_bg + bg_mask.shape[1]]

        # print('obj_before', obj_mask)
        # print('obj_crop', center_mask_obj)
        # print('')
        # print('bg_crop', center_mask_bg)
        # print('bg_before', bg_mask)


        obj_mask = torch.tensor(center_mask_obj, device=device).unsqueeze(0).unsqueeze(0)
        bg_mask = torch.tensor(center_mask_bg, device=device).unsqueeze(0).unsqueeze(0)

    return obj_mask, bg_mask

def split_features(feature_map, mask, split_thres=0.5, erosion_iters=0):

    obj_mask, bg_mask = split_mask(mask, split_thres, erosion_iters)

    # Set object template features
    inds = obj_mask.nonzero().transpose(0, 1)
    obj_features = feature_map[:, inds[2], inds[3]]
    # Size: (c, m0)

    # Set background template features
    inds = bg_mask.nonzero().transpose(0, 1)
    bg_features = feature_map[:, inds[2], inds[3]]
    # Size: (c, m1)

    return obj_features, bg_features


def compute_similarity(cur_feature, feature_templates, shape_s, shape_l, topk=20):

    similarity_scores = cur_feature.matmul(feature_templates) # Size: (h*w, m0)
    # print(similarity_scores.max(), similarity_scores.min())
    topk_scores = similarity_scores.topk(k=topk, dim=1, largest=True)
    avg_scores = topk_scores.values.mean(dim=1).reshape(1, 1, shape_s[0], shape_s[1])
    scores = F.interpolate(avg_scores, shape_l, mode='bilinear', align_corners=False)

    return scores


def adjust_rates(idx, l0, l1, l2):

    if idx % 10 == 0:
        l0 *= 0.9
    l1 = 1 - l0 - l2

    return l0, l1, l2

def eval_AANetNet():
    
    torch.set_printoptions(precision=3, threshold=30000, linewidth=160, sci_mode=False)
    np.set_printoptions(precision=3, threshold=30000, linewidth=160, suppress=False)

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
        '--no-temporal', action='store_true',
        help='Evaluate the video without temporally updating templates.')
    parser.add_argument(
        '--no-conf', action='store_true',
        help='Evaluate the video without high-confidence features updating templates.')
    parser.add_argument(
        '--no-aa', action='store_true',
        help='Evaluate the video without appearance-adaptive branch.')
    parser.add_argument(
        '--sample', action='store_true',
        help='The video sequence has been sampled.')
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show segmentation results from different stages.')
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
        '-o', '--out-folder', default=os.path.join(cfg['paths'][cfg_dataset], 'results'), type=str, metavar='PATH',
        help='Folder for the output segmentations.')
    args = parser.parse_args()

    print('Args:', args)

    if args.checkpoint is None:
        raise ValueError('Must input checkpoint path.')
    if args.video_name is None:
        raise ValueError('Must input video name.')
    
    # Hyper parameters 2
    water_thres = 0.5
    l0, l1, l2 = 0.4, 0.2, 0.4
    if args.no_conf:
        l0, l1, l2 = 0.67, 0.33, 0
    l_aa = 0.9

    # Dataset
    dataset_args = {}
    if torch.cuda.is_available():
        dataset_args = {
            'num_workers': 1, # int(cfg['params_AA']['num_workers']),
            'pin_memory': False # bool(cfg['params_AA']['pin_memory'])
        }

    dataset = WaterDataset_RGB(
        mode='eval',
        dataset_path=cfg['paths'][cfg_dataset], 
        test_case=args.video_name,
        eval_size=(int(cfg['params_AA']['eval_w']), int(cfg['params_AA']['eval_h']))
    )
    
    eval_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        **dataset_args
    )

     # Model
    feature_net = FeatureNet()
    deconv_net = DeconvNet()

    # Load pretrained model
    if os.path.isfile(args.checkpoint):
        print('Load checkpoint \'{}\''.format(args.checkpoint))
        if torch.cuda.is_available():
            checkpoint = torch.load(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
        args.start_epoch = checkpoint['epoch'] + 1
        feature_net.load_state_dict(checkpoint['feature_net'])
        deconv_net.load_state_dict(checkpoint['deconv_net'])
        print('Loaded checkpoint \'{}\' (epoch {})'
                .format(args.checkpoint, checkpoint['epoch']))
    else:
        raise ValueError('No checkpoint found at \'{}\''.format(args.checkpoint))

    # Set ouput path
    setting_prefix = args.model_name
    if args.no_temporal:
        setting_prefix += '_no_temporal'
    if args.no_conf:
        setting_prefix += '_no_conf'
    if args.no_aa:
        setting_prefix += '_no_aa'
    
    out_path = os.path.join(args.out_folder, setting_prefix + '_segs', args.video_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    if args.sample:
        out_full_path = out_path + '_full'
        if not os.path.exists(out_full_path):
            os.makedirs(out_full_path)

    # Start testing
    feature_net.to(device).eval()
    deconv_net.to(device).eval()
    running_time = AverageMeter()
    running_endtime = time.time()
    
    # Get the first frame annotation
    first_frame_mask = dataset.get_first_frame_label()
    first_frame_seg = TF.to_pil_image(first_frame_mask)
    first_frame_seg.save(os.path.join(out_path, '0.png'))
    if args.sample:
        first_frame_seg.save(os.path.join(out_full_path, '0.png'))
    first_frame_mask = first_frame_mask.to(device).unsqueeze(0)

    # Get the first frame features
    first_frame = dataset.get_first_frame().to(device).unsqueeze(0)
    eval_size = first_frame.shape[-2:]
    f3, f0, f1, f2 = feature_net(first_frame)
    f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
    f3 = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=False)
    feature0 = torch.cat((f1, f2, f3), 1)

    # Normalize features. Size: (c, h, w)
    feature_map = feature0.detach().squeeze(0)
    feature_norms = feature_map.norm(p=2, dim=0, keepdim=True)
    feature_map = feature_map / feature_norms
    c, h, w = feature_map.shape
    feature_n = h * w

    print('c', c, 'h', h, 'w', w)

    pre_frame_mask = F.interpolate(first_frame_mask, size=(h, w), mode='bilinear', align_corners=False).detach()
    # pre_frame_mask Size (1, 1, h, w)
    t_obj_first, t_bg_first = split_features(feature_map, pre_frame_mask)
    t_obj_temporal, t_bg_temporal = t_obj_first, t_bg_first
    t_temporal_n = [(t_obj_temporal.shape[1], t_bg_temporal.shape[1])]

    print('First frame features.\t', 'obj:', t_obj_first.shape, 'bg:', t_bg_first.shape)

    keep_features_n = int(cfg['params_AA']['temporal_n'])

    print('Erosion params.\t', 'Conf:', int(cfg['params_AA']['r0']), 'Temporal:', int(cfg['params_AA']['r1']))
    print('\n')

    with torch.no_grad():
        for i, sample in enumerate(tqdm(eval_loader)):
            
            img = sample['img'].to(device)     

            f3, f0, f1, f2 = feature_net(img)
            seg_fcn = deconv_net(f3, f0, f1, f2, img.shape).detach() # Size: (1, 1, h, w)

            f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            f3 = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=False)
            feature_map = torch.cat((f1, f2, f3), 1)

            l0, l1, l2 = adjust_rates(i + 1, l0, l1, l2)
            # print(i, l0, l1, l2)

            if not args.no_aa:

                feature_map = feature_map.detach().squeeze(0)
                feature_map /= feature_map.norm(p=2, dim=0, keepdim=True) # Size: (c, h, w)
                cur_feature = feature_map.reshape((c, feature_n)).transpose(0, 1)

                # Seg first frame features
                scores_obj = compute_similarity(cur_feature, t_obj_first, (h,w), img.shape[2:], topk=int(cfg['params_AA']['topk']))
                scores_bg = compute_similarity(cur_feature, t_bg_first, (h,w), img.shape[2:], topk=int(cfg['params_AA']['topk']))
                seg_first = ((scores_obj - scores_bg + 2) / 4).squeeze(1)
                val_min = seg_first.min()
                seg_first = (seg_first - val_min) / (seg_first.max() - val_min)

                # Seg temporal features 
                if t_obj_temporal.shape[1] > int(cfg['params_AA']['topk']):
                    scores_obj = compute_similarity(cur_feature, t_obj_temporal, (h,w), img.shape[2:], topk=int(cfg['params_AA']['topk']))
                else:
                    scores_obj = torch.zeros(img.shape[2:]).to(device)
                
                if t_bg_temporal.shape[1] > int(cfg['params_AA']['topk']):
                    scores_bg = compute_similarity(cur_feature, t_bg_temporal, (h,w), img.shape[2:], topk=int(cfg['params_AA']['topk']))
                else:
                    scores_bg = torch.zeros(img.shape[2:]).to(device)

                # For visualization
                # if i == 24:
                #     print(scores_obj.shape, scores_bg.shape)
                #     tmp_img = TF.to_pil_image(scores_obj.squeeze(0).cpu())
                #     tmp_img.save(f'tmp/obj_{i}.png')
                #     tmp_img = TF.to_pil_image(scores_bg.squeeze(0).cpu())
                #     tmp_img.save(f'tmp/bg_{i}.png')

                seg_temporal = ((scores_obj - scores_bg + 2) / 4).squeeze(1)
                val_min = seg_temporal.min()
                seg_temporal = (seg_temporal - val_min) / (seg_temporal.max() - val_min)

                if not args.no_conf:
                    # Add center features to template features
                    t_obj_conf, t_bg_conf = split_features(feature_map, pre_frame_mask, erosion_iters=int(cfg['params_AA']['r0']))
                    # print('Conf features.\t', 'obj:', t_obj_conf.shape, 'bg:', t_bg_conf.shape)
                    
                    if t_obj_conf.shape[1] > int(cfg['params_AA']['topk']):
                        scores_obj = compute_similarity(cur_feature, t_obj_conf, (h,w), img.shape[2:], topk=int(cfg['params_AA']['topk']))
                    else:
                        scores_obj = torch.zeros(img.shape[2:]).to(device)

                    if t_bg_conf.shape[1] > int(cfg['params_AA']['topk']):
                        scores_bg = compute_similarity(cur_feature, t_bg_conf, (h,w), img.shape[2:], topk=int(cfg['params_AA']['topk']))
                    else:
                        scores_bg = torch.zeros(img.shape[2:]).to(device)
                    
                    seg_conf = ((scores_obj - scores_bg + 2) / 4).squeeze(1)
                    val_min = seg_conf.min()
                    seg_conf = (seg_conf - val_min) / (seg_conf.max() - val_min)

                    seg_aa = l0 * seg_first + l1 * seg_temporal + l2 * seg_conf
                else:
                    seg_aa = l0 * seg_first + l1 * seg_temporal

                # seg_aa = torch.where(seg_aa > 0.5, one_tensor, zero_tensor)    
                seg_final = l_aa * seg_aa + (1 - l_aa) * seg_fcn
            else:
                seg_final = seg_fcn
            

            seg_final = torch.where(seg_final > 0.5, one_tensor, zero_tensor) # Size: (1, 1, h, w)
            pre_frame_mask = F.interpolate(seg_final, size=(h, w), mode='bilinear', align_corners=False).detach()
            
            if not args.no_aa:

                # Remove previous feature templates
                if i >= keep_features_n:
                    j = i - keep_features_n
                    t_obj_temporal = t_obj_temporal[:,t_temporal_n[j][0]:]
                    t_bg_temporal = t_bg_temporal[:,t_temporal_n[j][1]:]

                    # print('Removed old temporal templates.\t', t_obj_temporal.shape[1], t_bg_temporal.shape[1])

                # Add current features to template features
                cur_features_obj, cur_features_bg = split_features(feature_map, pre_frame_mask, split_thres=float(cfg['params_AA']['hc']), erosion_iters=int(cfg['params_AA']['r1']))
                # print('cur features:\t', 'obj:', cur_features_obj.shape, 'bg:', cur_features_bg.shape)
                t_obj_temporal = torch.cat((t_obj_temporal, cur_features_obj), dim=1)
                t_bg_temporal = torch.cat((t_bg_temporal, cur_features_bg), dim=1)

                t_temporal_n.append((cur_features_obj.shape[1], cur_features_bg.shape[1]))
                # print('temporal features:\t', 'obj:', t_obj_temporal.shape, 'bg:', t_bg_temporal.shape)

            # print('output', output)
            # print('adapt', adaption_seg)
            # print('final', seg_final)

            running_time.update(time.time() - running_endtime)
            running_endtime = time.time()

            # print('Segment: [{0:4}/{1:4}]\t'
            #     'Time: {running_time.val:.3f}s ({running_time.sum:.3f}s)\r'.format(
            #     i + 1, len(eval_loader), running_time=running_time))
            
            seg_final = TF.to_pil_image(seg_final.squeeze(0).cpu())
            if args.sample:
                seg_final.save(os.path.join(out_full_path, f'{i + 1}.png'))

                if i + 1 in [1, 50, 100, 150, 199]:
                    seg_final.save(os.path.join(out_path, f'{i + 1}.png'))        
            
            else:
                seg_final.save(os.path.join(out_path, f'{i + 1}.png'))        

            if args.verbose:
                
                fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))
                # for rows in axes:
                #     for ax in rows:
                #         ax.set_axis_off()

                seg_first = TF.to_pil_image(seg_first.squeeze(0).cpu())
                axes[0][0].imshow(seg_first, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
                axes[0][0].set_title('(a) seg by first frame')

                seg_temporal = TF.to_pil_image(seg_temporal.squeeze(0).cpu())
                axes[0][1].imshow(seg_temporal, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
                axes[0][1].set_title('(b) seg by temporal info')

                seg_conf = TF.to_pil_image(seg_conf.squeeze(0).cpu())
                axes[0][2].imshow(seg_conf, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
                axes[0][2].set_title('(c) seg by confidence area')

                seg_aa = TF.to_pil_image(seg_aa.squeeze(0).cpu())
                axes[1][0].imshow(seg_aa, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
                axes[1][0].set_title('(d) seg by water branch')

                seg_fcn = TF.to_pil_image(seg_fcn.squeeze(0).cpu())
                axes[1][1].imshow(seg_fcn, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
                axes[1][1].set_title('(e) seg by parent branch')

                axes[1][2].imshow(seg_final, cmap='gray', interpolation='nearest', vmin=0, vmax=255)
                axes[1][2].set_title('(f) final segmentation')

                if args.sample:
                    plt.savefig(os.path.join(out_full_path, 'v_%d.png' %(i + 1)), bbox_inches='tight', pad_inches = 0)
                else:
                    plt.savefig(os.path.join(out_path, 'v_%d.png' %(i + 1)), bbox_inches='tight', pad_inches = 0)

                plt.close(fig)

            
            

            # if args.benchmark:
            #     gt_seg = load_image_in_PIL(os.path.join(gt_folder, gt_list[i])).convert('L')
            #     gt_seg.thumbnail((int(cfg['params_AA']['eval_h']), int(cfg['params_AA']['eval_w'])), Image.ANTIALIAS)
            #     gt_tf = TF.to_tensor(gt_seg).to(device).type(torch.int)

            #     iou = iou_tensor(seg_final.squeeze(0).type(torch.int), gt_tf)
            #     avg_iou += iou.item()
            #     print('iou:', iou.item())



    # if args.benchmark:
    #     print('total_iou:', avg_iou)
    #     avg_iou /= len(eval_loader)
    #     print('avg_iou:', avg_iou, 'frame_num:', len(eval_loader))
        
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

    print('\n')

    if args.sample:
        mask_folder = args.video_name + '_full'
    else:
        mask_folder = args.video_name
    run_cvt_images_to_overlays(args.video_name, mask_folder, cfg['paths'][cfg_dataset], setting_prefix, eval_size)

    print('\n')
    
if __name__ == '__main__':
    eval_AANetNet()

