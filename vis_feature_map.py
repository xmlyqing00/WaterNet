import os
import argparse
import sys
import time
import numpy as np
import cv2
import torch
import configparser
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image

from src.network import VisFeatureMapNet
from src.AANet import ParentNet
from src.dataset import WaterDataset_RGB


def show_feature_map_similarity():
    
    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    if sys.platform == 'darwin':
        cfg_dataset = 'dataset_mac'
    elif sys.platform == 'linux':
        cfg_dataset = 'dataset_linux'

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch VisFeatureMapNet Testing')
    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    parser.add_argument(
        '-v', '--video-name', default=None, type=str,
        help='Test video name (default: none).')
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
            'num_workers': int(cfg['params']['num_workers']),
            'pin_memory': bool(cfg['params']['pin_memory'])
        }

    dataset = WaterDataset_RGB(
        mode='eval',
        dataset_path=cfg['paths'][cfg_dataset], 
        test_case=args.video_name
    )
    eval_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        **dataset_args
    )

    # Model
    VisFeatureMap_net = VisFeatureMapNet()
    Parent_net = Parent_net()

    # Load pretrained model
    if os.path.isfile(args.checkpoint):
        print('Load checkpoint \'{}\''.format(args.checkpoint))
        if torch.cuda.is_available():
            checkpoint = torch.load(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
        args.start_epoch = checkpoint['epoch'] + 1
        VisFeatureMap_net.load_state_dict(checkpoint['model'])
        print('Loaded checkpoint \'{}\' (epoch {})'
                .format(args.checkpoint, checkpoint['epoch']))
    else:
        raise ValueError('No checkpoint found at \'{}\''.format(args.checkpoint))

    # Set ouput path
    
    out_path = os.path.join(args.out_folder, 'visualization')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Start testing
    VisFeatureMap_net.to(device).eval()
    
    # Feature map 0
    test_id = 35
    sample = dataset[test_id]

    img = sample['img'].to(device).unsqueeze(0)     

    feature_map0 = VisFeatureMap_net(img).detach().squeeze(0).numpy()

    # Feature map 1
    test_id = 34
    sample = dataset[test_id]

    img = sample['img'].to(device).unsqueeze(0)     

    feature_map1 = VisFeatureMap_net(img).detach().squeeze(0).numpy()

    # Feature diff
    c, h, w = feature_map0.shape
    print(c, h, w)

    # Position
    x, y = 20, 20

    fig, axes = plt.subplots(nrows=1, ncols=2)

    diff_map = (feature_map0 - feature_map1) ** 2
    diff_map = np.sqrt(np.sum(diff_map, 0) / c)
    axes[0].imshow(diff_map, cmap='plasma', interpolation='nearest')
    
    vec = feature_map1[:,y,x]
    vec_tile = np.tile(vec, h * w).reshape(h, w, c).transpose(2, 0, 1)

    diff_map = (feature_map1 - vec_tile) ** 2
    diff_map = np.sqrt(np.sum(diff_map, 0) / c)
    im = axes[1].imshow(diff_map, cmap='plasma', interpolation='nearest')
    
    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()

    
if __name__ == '__main__':
    show_feature_map_similarity()
