import os
import argparse
import sys
import time
import cv2
import torch
import configparser
from torch.utils import model_zoo
from torchvision import transforms

from src.network import MaskTrackNet
from src.dataset import WaterDataset
from src.avg_meter import AverageMeter


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
        '-o', '--out-path', default=os.path.join(cfg['paths']['dataset'], 'MaskTrackNet_segs/'), type=str, metavar='PATH',
        help='Path to the output segmentations (default: none).')
    args = parser.parse_args()

    print('Args:', args)

    if args.checkpoint is None:
        raise ValueError('Must input checkpoint path.')
    if args.video_name is None:
        raise ValueError('Must input video name.')

    water_thres = 5

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
        dataset_path=cfg['path']['dataset_path'], 
        test_case=args.video_name
    )
    test_loader = torch.utils.data.DataLoader(
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
    out_path = os.path.join(args.out_path, args.video_name)
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    # Start testing
    mt_net.eval()

    # First frame annotation
    first_frame_label = dataset.get_first_frame_label()

    for i, sample in enumerate(test_loader):
        
        print('Segment: [{0:4}/{1:4}'.format(i, len(test_loader)))

        img = sample['img'].to(device)        
        output = mt_net(input)

        seg = output.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
        ret, seg = cv2.threshold(seg, water_thres, 255, cv2.THRESH_BINARY)

        seg_path = os.path.join(args.out_path, str(i) + '.png')
        cv2.imwrite(seg_path, seg)


if __name__ == '__main__':
    test_FCNResNet()
