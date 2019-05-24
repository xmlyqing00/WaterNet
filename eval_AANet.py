import os
import argparse
import sys
import time
import configparser
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage

from src.AANet import FeatureNet, DeconvNet
from src.dataset import WaterDataset_RGB
from src.avg_meter import AverageMeter
from src.cvt_images_to_overlays import run_cvt_images_to_overlays

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

one_tensor = torch.ones(1).to(device)
zero_tensor = torch.zeros(1).to(device)


def split_mask(mask, split_thres=0.7, erosion_iters=0):

    obj_mask = torch.where(mask > split_thres, one_tensor, zero_tensor)
    bg_mask = torch.where(mask < (1-split_thres), one_tensor, zero_tensor)

    if erosion_iters > 0:
        obj_mask = obj_mask.squeeze().cpu().numpy()
        bg_mask = bg_mask.squeeze().cpu().numpy()
        
        center_mask_obj = ndimage.binary_erosion(obj_mask, iterations=erosion_iters).astype(np.float32)
        center_mask_bg = ndimage.binary_erosion(bg_mask, iterations=erosion_iters).astype(np.float32)
        
        # Debug
        # print('obj_mask', obj_mask)
        # print('center_mask_obj', center_mask_obj)

        obj_mask = torch.tensor(center_mask_obj, device=device).unsqueeze(0).unsqueeze(0)
        bg_mask = torch.tensor(center_mask_bg, device=device).unsqueeze(0).unsqueeze(0)

    return obj_mask, bg_mask

def split_features(feature_map, mask, split_thres=0.7, erosion_iters=0):

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


def compute_similarity(cur_feature, template_features, shape_s, shape_l, topk=20):

    similarity_scores = cur_feature.matmul(template_features) # Size: (h*w, m0)
    topk_scores = similarity_scores.topk(k=topk, dim=1, largest=True)
    avg_scores = topk_scores.values.mean(dim=1).reshape(1, 1, shape_s[0], shape_s[1])
    scores = F.interpolate(avg_scores, shape_l, mode='bilinear', align_corners=False)

    return scores

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
            'num_workers': int(cfg['params_AA']['num_workers']),
            'pin_memory': bool(cfg['params_AA']['pin_memory'])
        }

    dataset = WaterDataset_RGB(
        mode='eval',
        dataset_path=cfg['paths'][cfg_dataset], 
        test_case=args.video_name,
        # eval_size=(640, 640)
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
    out_path = os.path.join(args.out_folder, args.model_name + '_segs', args.video_name)
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
    feature0, _, _, _ = feature_net(first_frame)

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
    template_features_obj, template_features_bg = split_features(feature_map, pre_frame_mask)
    m0 = template_features_obj.shape[1]
    m1 = template_features_bg.shape[1]

    print('Init features', template_features_obj.shape, template_features_bg.shape)

    with torch.no_grad():
        for i, sample in enumerate(eval_loader):

            img = sample['img'].to(device)     

            feature_map, f0, f1, f2 = feature_net(img)
            output = deconv_net(feature_map, f0, f1, f2, img.shape).detach() # Size: (1, 1, h, w)

            feature_map = feature_map.detach().squeeze(0)
            feature_map /= feature_map.norm(p=2, dim=0, keepdim=True) # Size: (c, h, w)

            # Add center features to template features
            center_features_obj, center_features_bg = split_features(feature_map, pre_frame_mask, erosion_iters=4)
            print('Conf features:\t', center_features_obj.shape, center_features_bg.shape)
            template_features_obj = torch.cat((template_features_obj, center_features_obj), dim=1)
            template_features_bg = torch.cat((template_features_bg, center_features_bg), dim=1)
            print('Added conf features:\t', template_features_obj.shape, template_features_bg.shape)

            cur_feature = feature_map.reshape((c, feature_n)).transpose(0, 1)
            scores_obj = compute_similarity(cur_feature, template_features_obj, (h,w), img.shape[2:])
            scores_bg = compute_similarity(cur_feature, template_features_bg, (h,w), img.shape[2:])
            scores = torch.cat((scores_bg, scores_obj), dim=1)
            adaption_seg = F.softmax(scores, dim=1).argmax(dim=1).type(torch.float).detach() # Size: (1, 1, h, w)

            final_seg = (output + adaption_seg) / 2
            final_seg = torch.where(final_seg > 0.5, one_tensor, zero_tensor) # Size: (1, 1, h, w)
            pre_frame_mask = F.interpolate(final_seg, size=(h, w), mode='bilinear', align_corners=False).detach()

            # Update templates
            template_features_obj = template_features_obj[:,:m0]
            template_features_bg = template_features_bg[:,:m1]
            print('Removed conf features:\t', template_features_obj.shape, template_features_bg.shape)

            # print('output', output)
            # print('adapt', adaption_seg)
            # print('final', final_seg)

            seg0 = TF.to_pil_image(output.squeeze(0).cpu())
            seg0.save(os.path.join(out_path, 'seg0_%d.png' % (i + 1)))

            seg1 = TF.to_pil_image(adaption_seg.squeeze(0).cpu())
            seg1.save(os.path.join(out_path, 'seg1_%d.png' % (i + 1)))

            final_seg = TF.to_pil_image(final_seg.squeeze(0).cpu())
            final_seg.save(os.path.join(out_path, '%d.png' % (i + 1)))

            running_time.update(time.time() - running_endtime)
            running_endtime = time.time()

            print('Segment: [{0:4}/{1:4}]\t'
                'Time: {running_time.val:.3f}s ({running_time.sum:.3f}s)\t'.format(
                i + 1, len(eval_loader), running_time=running_time))


    run_cvt_images_to_overlays(args.video_name, args.out_folder, args.model_name, eval_size)
    
if __name__ == '__main__':
    eval_AANetNet()

