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


def test_FCNResNet():
    
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
        '-o', '--out-path', default=None, type=str, metavar='PATH',
        help='Path to the output segmentations (default: none).')
    args = parser.parse_args()

    print('Args:', args)

    if args.checkpoint is None:
        raise ValueError('Must input checkpoint path.')
    if args.imgs_path is None:
        raise ValueError('Must input test images path.')
    if args.out_path is None:
        raise ValueError('Must input output images path.')

    water_thres = 5

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Dataset
    dataset_args = {}
    if torch.cuda.is_available():
        dataset_args = {
            'num_workers': 4,
            'pin_memory': True
        }

    imagenet_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    dataset = WaterDataset(
        mode='test',
        dataset_path=cfg['path']['dataset_path'], 
        img_transforms=transforms.Compose([
            transforms.ToTensor(),
            imagenet_normalize
        ])
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

    # Start testing
    
    mt_net.eval()

    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)

    for i, input in enumerate(test_loader):
        
        print(i)

        input = input.to(device)
        output = mt_net(input)

        seg = output.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
        ret, seg = cv2.threshold(seg, water_thres, 255, cv2.THRESH_BINARY)

        seg_path = os.path.join(args.out_path, str(i) + '.png')
        cv2.imwrite(seg_path, seg)


if __name__ == '__main__':
    test_FCNResNet()
