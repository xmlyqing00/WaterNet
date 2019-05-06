import os
import argparse
import sys
import time
import cv2
import torch
import configparser
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

from src.network import PureFeatureDetectorNet
from src.dataset import WaterDataset_PFD
from src.avg_meter import AverageMeter
from src.cvt_images_to_overlays import run_cvt_images_to_overlays


def minmax_normalize(data_tensor):

    min_val = torch.min(data_tensor)
    max_val = torch.max(data_tensor)
    data_tensor = (data_tensor - min_val) / (max_val - min_val)

    return data_tensor


def eval_PFDNet():
    
    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch RGBMaskNet Testing')
    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    parser.add_argument(
        '-i', '--video-name', default=None, type=str,
        help='Test video name (default: none).')
    parser.add_argument(
        '-m', '--model-name', default='PFDNet_segs', type=str,
        help='Model name for the ouput segmentation, it will create a subfolder under the out_folder (default: none).')
    parser.add_argument(
        '-o', '--out-folder', default=cfg['paths']['dataset'], type=str, metavar='PATH',
        help='Folder for the output segmentations.')
    args = parser.parse_args()

    print('Args:', args)

    if args.checkpoint is None:
        raise ValueError('Must input checkpoint path.')
    if args.video_name is None:
        raise ValueError('Must input video name.')

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Dataset
    dataset_args = {}
    if torch.cuda.is_available():
        dataset_args = {
            'num_workers': int(cfg['params']['num_workers']),
            'pin_memory': bool(cfg['params']['pin_memory'])
        }

    dataset = WaterDataset_PFD(
        mode='eval',
        dataset_path=cfg['paths']['dataset'], 
        test_case=args.video_name
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        **dataset_args
    )

    # Model
    pfd_net = PureFeatureDetectorNet().to(device)

    # Load pretrained model
    if os.path.isfile(args.checkpoint):
        print('Load checkpoint \'{}\''.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        args.start_epoch = checkpoint['epoch'] + 1
        pfd_net.load_state_dict(checkpoint['model'])
        print('Loaded checkpoint \'{}\' (epoch {})'
                .format(args.checkpoint, checkpoint['epoch']))
    else:
        raise ValueError('No checkpoint found at \'{}\''.format(args.checkpoint))

    # Set ouput path
    
    out_path = os.path.join(args.out_folder, args.model_name, args.video_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Start testing
    pfd_net.eval()
    running_time = AverageMeter()
    running_endtime = time.time()
    
    # First frame annotation
    pre_frame_mask = dataset.get_first_frame_label()
    first_frame_seg = TF.to_pil_image(pre_frame_mask)
    first_frame_seg.save(os.path.join(out_path, '0.png'))
    pre_frame_mask = pre_frame_mask.unsqueeze(0).to(device)

    black_tensor1 = TF.to_tensor(np.zeros(pre_frame_mask.shape, np.uint8))
    black_tensor3 = TF.to_tensor(np.zeros(pre_frame_mask.shape, np.uint8))

    for i, sample in enumerate(eval_loader):
        
        img = sample['img'].to(device)  

        label_water = pre_frame_mask == 1
        mask = torch.cat((label_water, label_water, label_water), 0)
        img_water = torch.where(mask, img, black_tensor3)

        mask = ~mask
        img_bg = torch.where(mask, img, black_tensor3)

        img = torch.cat((img_water, img_bg), 1)

        output = pfd_net(img)

        pre_frame_mask = minmax_normalize(output.detach())
        seg_raw = TF.to_pil_image(pre_frame_mask.squeeze(0).cpu())
        seg_raw.save(os.path.join(out_path, 'raw_%d.png' % (i + 1)))

        zero_tensor = torch.zeros(pre_frame_mask.shape).to(device)
        one_tensor = torch.ones(pre_frame_mask.shape).to(device)
        # pre_frame_mask = torch.where(pre_frame_mask > water_thres, pre_frame_mask, zero_tensor)
        seg = TF.to_pil_image(pre_frame_mask.squeeze(0).cpu())
        seg.save(os.path.join(out_path, '%d.png' % (i + 1)))

        running_time.update(time.time() - running_endtime)
        running_endtime = time.time()

        print('Segment: [{0:4}/{1:4}]\t'
            'Time: {running_time.val:.0f}s ({running_time.sum:.0f}s)\t'.format(
            i + 1, len(eval_loader), running_time=running_time))

    run_cvt_images_to_overlays(args.video_name, args.model_name)
    
if __name__ == '__main__':
    eval_PFDNet()
