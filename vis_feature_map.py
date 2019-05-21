import os
import argparse
import sys
import time
import numpy as np
import cv2
import torch
from torch.utils import model_zoo
import configparser
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from PIL import Image

from src.AANet import FeatureNet
from src.dataset import WaterDataset_RGB


def vis_features(feature_map0, feature_map1, x, y):

    # Feature diff
    c, h, w = feature_map0.shape
    print(c, h, w)

    x = int(x * w)
    y = int(y * h)

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
    test_id = 38
    sample = dataset[test_id]

    img = sample['img'].to(device).unsqueeze(0)     

    feature_map2, t0, t1, t2 = feature_net(img)
    feature_map2 = feature_map2.detach().squeeze(0).cpu().numpy()
    # print(t0.shape)
    # print(t1.shape)
    # print(t2.shape)

    # Feature map 1
    test_id = test_id + 1
    sample = dataset[test_id]

    img = sample['img'].to(device).unsqueeze(0)     

    feature_map3, _, _, _ = feature_net(img)
    feature_map3 = feature_map3.detach().squeeze(0).cpu().numpy()

    # Position
    x, y = 0.5, 0.7
    vis_features(feature_map2, feature_map3, x, y)
    
if __name__ == '__main__':
    show_feature_map_similarity()
