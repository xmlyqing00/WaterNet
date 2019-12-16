import os
import argparse
import sys
import time
import numpy as np
import cv2
import torch
import random
from torch.utils import model_zoo
import configparser
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image

from src.WaterNet import FeatureNet
from src.dataset import WaterDataset_RGB

def diff_feature_map(feature_map0, feature_map1):

    c, h, w = feature_map0.shape
    print('c h w:', c, h, w)

    fig, axes = plt.subplots(nrows=1, ncols=1)

    diff_map = (feature_map0 - feature_map1) ** 2
    diff_map = np.sqrt(np.sum(diff_map, 0) / c)
    im = axes[0].imshow(diff_map, cmap='plasma', interpolation='nearest', vmin=0, vmax=2.5)
    
    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()

def vis_features(feature_map, xy_list):

    # Feature diff
    c, h, w = feature_map.shape
    print('c h w:', c, h, w)

    fig, axes = plt.subplots(nrows=1, ncols=3)

    diff_map_min = None

    for i, xy in enumerate(xy_list):
        x = int(xy[0] * w)
        y = int(xy[1] * h)

        print('x y:', x, y)

        vec = feature_map[:,y,x]
        vec_tile = np.tile(vec, h * w).reshape(h, w, c).transpose(2, 0, 1)

        diff_map = (feature_map - vec_tile) ** 2
        diff_map = np.sqrt(np.sum(diff_map, 0) / c)

        if i == 0:
            diff_map_min = diff_map
        else:
            diff_map_min = np.minimum(diff_map_min, diff_map)

        if i == 0:    
            im = axes[0].imshow(diff_map_min, cmap='plasma', interpolation='nearest', vmin=0, vmax=2.5)
        elif i == int(len(xy_list)/4) - 1:
            im = axes[1].imshow(diff_map_min, cmap='plasma', interpolation='nearest', vmin=0, vmax=2.5)
        elif i == len(xy_list) - 1:
            im = axes[2].imshow(diff_map_min, cmap='plasma', interpolation='nearest', vmin=0, vmax=2.5)

    fig.colorbar(im, ax=axes.ravel().tolist())

    plt.show()


def show_feature_map_similarity():
    
    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    if sys.platform == 'darwin':
        cfg_dataset = 'dataset_mac'
    elif sys.platform == 'linux':
        cfg_dataset = 'dataset_ubuntu'

    # Hyper parameters
    parser = argparse.ArgumentParser(description='PyTorch Feature Net Visualization')
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

    if args.video_name is None:
        raise ValueError('Must input video name.')

    water_thres = 0.5

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Dataset
    dataset = WaterDataset_RGB(
        mode='eval',
        dataset_path=cfg['paths'][cfg_dataset], 
        test_case=args.video_name,
        eval_size=(640, 640)
    )

    # Model
    feature_net = FeatureNet()

    # Load pretrained model
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            print('Load checkpoint \'{}\''.format(args.checkpoint))
            if torch.cuda.is_available():
                checkpoint = torch.load(args.checkpoint)
            else:
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
            args.start_epoch = checkpoint['epoch'] + 1
            feature_net.load_state_dict(checkpoint['feature_net'])
            print('Loaded checkpoint \'{}\' (epoch {})'
                    .format(args.checkpoint, checkpoint['epoch']))
        else:
            raise ValueError('No checkpoint found at \'{}\''.format(args.checkpoint))
    else:    
        print('Load pretrained ResNet 34.')
        resnet34_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
        pretrained_model = model_zoo.load_url(resnet34_url)
        feature_net.load_pretrained_model(pretrained_model)


    # Set ouput path
    
    out_path = os.path.join(args.out_folder, 'visualization')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Start testing
    feature_net.to(device).eval()
    
    # Feature map 0
    test_id = 28
    sample = dataset[test_id]

    with torch.no_grad():
        img = sample['img'].to(device).unsqueeze(0)
        feature_map0, _, _, _ = feature_net(img)
        feature_map0 = feature_map0.detach().squeeze(0).cpu().numpy()

    # Feature map 1
    test_id = test_id + 1
    sample = dataset[test_id]

    with torch.no_grad():
        img = sample['img'].to(device).unsqueeze(0)     
        feature_map1, _, _, _ = feature_net(img)
        feature_map1 = feature_map1.detach().squeeze(0).cpu().numpy()

    # diff_feature_map(feature_map0, feature_map1)

    # Position
    # xy_list = [
    #     (0.35, 0.65625), (0.35, 0.4375), (0.325, 0.375), (0.675, 0.6875), (0.5, 0.7), 
    #     (0.375, 0.84375), (0.15, 0.875), (0.65, 0.53125), (0.325, 0.90625), (0.85, 0.84375)]
    xy_list = []
    for i in range(20):
        x = random.random() * 0.8 + 0.1
        y = random.random() * 0.425 + 0.375
        xy_list.append((x,y))

    print(xy_list)
    vis_features(feature_map1, xy_list)

    
if __name__ == '__main__':
    show_feature_map_similarity()
