import os
import argparse
import sys
import time
import cv2
import torch
import configparser
import torchvision.transforms.functional as TF
from PIL import Image

from src.network import MaskTrackNet
from src.dataset import WaterDataset
from src.avg_meter import AverageMeter


def minmax_normalize(data_tensor):

    min_val = torch.min(data_tensor)
    max_val = torch.max(data_tensor)
    data_tensor = (data_tensor - min_val) / (max_val - min_val)

    return data_tensor


def eval_MaskTrackNet():
    
    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch MaskResNet Testing')
    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    parser.add_argument(
        '-i', '--video-name', default=None, type=str,
        help='Test video name (default: none).')
    parser.add_argument(
        '-m', '--model-name', default='MaskTrackNet_segs', type=str,
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

    water_thres = 0.8

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

    dataset = WaterDataset(
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
    mt_net = MaskTrackNet().to(device)

    # Load pretrained model
    if os.path.isfile(args.checkpoint):
        print('Load checkpoint \'{}\''.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        args.start_epoch = checkpoint['epoch'] + 1
        mt_net.load_state_dict(checkpoint['model'])
        print('Loaded checkpoint \'{}\' (epoch {})'
                .format(args.checkpoint, checkpoint['epoch']))
    else:
        raise ValueError('No checkpoint found at \'{}\''.format(args.checkpoint))

    # Set ouput path
    
    if not os.path.exists(args.out_folder):
        os.mkdir(args.out_folder)
    out_path = os.path.join(args.out_folder, args.model_name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    out_path = os.path.join(out_path, args.video_name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Start testing
    mt_net.eval()
    running_time = AverageMeter()
    running_endtime = time.time()
    
    # First frame annotation
    pre_frame_mask = dataset.get_first_frame_label()
    first_frame_seg = TF.to_pil_image(pre_frame_mask)
    first_frame_seg.save(os.path.join(out_path, '0.png'))
    pre_frame_mask = pre_frame_mask.unsqueeze(0).to(device)

    for i, sample in enumerate(eval_loader):
        
        img = sample['img'].to(device)     
        img_mask = torch.cat([img, pre_frame_mask], 1)  

        output = mt_net(img_mask)

        pre_frame_mask = minmax_normalize(output.detach())
        seg_raw = TF.to_pil_image(pre_frame_mask.squeeze(0).cpu())
        seg_raw.save(os.path.join(out_path, 'raw_%d.png' % (i + 1)))

        zero_tensor = torch.zeros(pre_frame_mask.shape).to(device)
        one_tensor = torch.ones(pre_frame_mask.shape).to(device)
        pre_frame_mask = torch.where(pre_frame_mask > water_thres, pre_frame_mask, zero_tensor)
        seg = TF.to_pil_image(pre_frame_mask.squeeze(0).cpu())
        seg.save(os.path.join(out_path, '%d.png' % (i + 1)))

        running_time.update(time.time() - running_endtime)
        running_endtime = time.time()

        print('Segment: [{0:4}/{1:4}]\t'
            'Time: {running_time.val:.3f}s ({running_time.sum:.3f}s)\t'.format(
            i + 1, len(eval_loader), running_time=running_time))


if __name__ == '__main__':
    eval_MaskTrackNet()
