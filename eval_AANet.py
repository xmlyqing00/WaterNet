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

from src.AANet import FeatureNet, DeconvNet
from src.dataset import WaterDataset_RGB
from src.avg_meter import AverageMeter
from src.cvt_images_to_overlays import run_cvt_images_to_overlays


def eval_AANetNet():
    
    torch.set_printoptions(precision=3, threshold=2000, sci_mode=False)

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

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

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
        eval_size=(640, 640)
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
    
    # First frame annotation
    first_frame_mask = dataset.get_first_frame_label()
    first_frame_seg = TF.to_pil_image(first_frame_mask)
    first_frame_seg.save(os.path.join(out_path, '0.png'))
    first_frame_mask = first_frame_mask.to(device).unsqueeze(0)

    # Init template features
    first_frame = dataset.get_first_frame().to(device).unsqueeze(0)
    feature0, _, _, _ = feature_net(first_frame)

    _, c, h, w = feature0.shape
    feature_n = h * w

    print(first_frame_mask.shape)
    first_frame_mask = F.interpolate(first_frame_mask, size=(h, w), mode='bilinear', align_corners=False)
    print(first_frame_mask.shape)

    print(first_frame_mask)
    return

    template_features = feature0.detach().reshape((c, feature_n))
    feature_norms = template_features.norm(2, 0, keepdim=True)
    template_features = template_features / feature_norms
    # Size: (c, h * w)

    for i, sample in enumerate(eval_loader):

        img = sample['img'].to(device)     

        feature_map, f0, f1, f2 = feature_net(img)
        output = deconv_net(feature_map, f0, f1, f2, img.shape)

        cur_feature = feature_map.detach().reshape((c, feature_n)).transpose(0, 1)
        feature_norms = cur_feature.norm(2, 1, keepdim=True)
        cur_feature /= feature_norms
        # Size: h*w, c

        similarity_scores = cur_feature.matmul(template_features)
        print(similarity_scores)
        topk_scores = similarity_scores.topk(20, 1, largest=True)
        print(topk_scores)

        # seg0 = TF.to_pil_image(output.detach().squeeze(0).cpu())
        # seg0 = dataset.resize_to_origin(seg0)
        # seg0.save(os.path.join(out_path, 'seg0_%d.png' % (i + 1)))

        running_time.update(time.time() - running_endtime)
        running_endtime = time.time()

        print('Segment: [{0:4}/{1:4}]\t'
            'Time: {running_time.val:.3f}s ({running_time.sum:.3f}s)\t'.format(
            i + 1, len(eval_loader), running_time=running_time))

    # run_cvt_images_to_overlays(args.video_name, args.out_folder, args.model_name)
    
if __name__ == '__main__':
    eval_AANetNet()
