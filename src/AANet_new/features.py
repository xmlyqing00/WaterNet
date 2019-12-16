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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils import model_zoo
from PIL import Image
from scipy import ndimage

from src.WaterNet.feature_net import FeatureNet
from src.WaterNet.dataset import WaterDataset


device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

class FeatureTemplates:

    def __init__(self, checkpoint=None):

        # Paths
        self.cfg = configparser.ConfigParser()
        self.cfg.read('settings.conf')

        self.feature_net = FeatureNet().to(device)

        if checkpoint:
            if os.path.isfile(checkpoint):
                print('Load checkpoint \'{}\''.format(checkpoint))
                checkpoint = torch.load(checkpoint)
                start_epoch = checkpoint['epoch'] + 1
                self.feature_net.load_state_dict(checkpoint['feature_net'])
                # feature_net_optimizer.load_state_dict(checkpoint['feature_net_optimizer'])
                print('Loaded checkpoint \'{}\' (epoch {})'
                    .format(checkpoint, checkpoint['epoch']))
            else:
                print('No checkpoint found at \'{}\''.format(checkpoint))
        else:
            print('Load pretrained ResNet 34.')
            resnet34_url = 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
            pretrained_model = model_zoo.load_url(resnet34_url)
            self.feature_net.load_pretrained_model(pretrained_model)

    def build_from_multiimages(self, dataset):

        dataset_args = {}
        if torch.cuda.is_available():
            dataset_args = {
                'num_workers': int(self.cfg['params_AA']['num_workers']),
                'pin_memory': bool(self.cfg['params_AA']['pin_memory'])
            }

        dataset_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=int(self.cfg['params_AA']['batch_size']),
            shuffle=True,
            **dataset_args
        )
    
        with torch.no_grad():
            for i, sample in enumerate(dataset_loader):

                img, label = sample['img'].to(device), sample['label'].to(device)
            
                f0, f1, f2, f3 = self.feature_net(img)

                f2 = F.interpolate(f2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
                f3 = F.interpolate(f3, size=f1.shape[-2:], mode='bilinear', align_corners=False)
                feature_map = torch.cat((f1, f2, f3), 1)
                feature_map = feature_map
                feature_map /= feature_map.norm(p=2, dim=1, keepdim=True) # Size: (b, c, h, w)
                b, c, h, w = feature_map.shape

                frame_mask = F.interpolate(label, size=(h, w), mode='bilinear', align_corners=False)

                f_obj, f_bg = split_features(feature_map, frame_mask)
                print(f_obj.shape, f_bg.shape)

                return f_obj, f_bg


    def build_from_dataset(self):
        
        dataset = WaterDataset(
            mode='dataset',
            dataset_path=self.cfg['paths']['dataset_ubuntu'],
            input_size=(int(self.cfg['params_AA']['input_w']), int(self.cfg['params_AA']['input_h']))
        )
        
        feature_dir = 'features/dataset/'
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        f_obj, f_bg = self.build_from_multiimages(dataset)
        torch.save(f_obj, os.path.join(feature_dir, 'f_obj.pt'))
        torch.save(f_bg, os.path.join(feature_dir, 'f_bg.pt'))
    
    def build_from_addon(self):

        dataset = WaterDataset(
            mode='addon',
            dataset_path=self.cfg['paths']['dataset_ubuntu'],
            input_size=(int(self.cfg['params_AA']['input_w']), int(self.cfg['params_AA']['input_h']))
        )

        feature_dir = 'features/addon/'
        if not os.path.exists(feature_dir):
            os.makedirs(feature_dir)

        f_obj, f_bg = self.build_from_multiimages(dataset)
        torch.save(f_obj, os.path.join(feature_dir, 'f_obj.pt'))
        torch.save(f_bg, os.path.join(feature_dir, 'f_bg.pt'))

        return f_obj, f_bg

    def build_from_eval(self, dataset):
        
        first_frame_mask = dataset.get_first_frame_label()
        first_frame_seg = TF.to_pil_image(first_frame_mask)

        return f_obj, f_bg


def split_mask(mask, split_thres):

    one_tensor = torch.ones(1).to(device)
    zero_tensor = torch.zeros(1).to(device)

    obj_mask = torch.where(mask > split_thres, one_tensor, zero_tensor)
    bg_mask = torch.where(mask < (1-split_thres), one_tensor, zero_tensor)

    return obj_mask, bg_mask

def split_features(feature_map, mask, split_thres=0.5):

    obj_mask, bg_mask = split_mask(mask, split_thres)
    print('obj', obj_mask.shape, 'bg', bg_mask.shape)

    # Set object template features
    inds = obj_mask.nonzero().transpose(0, 1)
    obj_features = feature_map[inds[0], :, inds[2], inds[3]]
    # Size: (c, m0)

    # Set background template features
    inds = bg_mask.nonzero().transpose(0, 1)
    bg_features = feature_map[inds[0], :, inds[2], inds[3]]
    # Size: (c, m1)

    return obj_features, bg_features

def calc_similarity(cur_feature, feature_templates, shape_s, shape_l, topk=20):

    similarity_scores = cur_feature.matmul(feature_templates) # Size: (h*w, m0)
    topk_scores = similarity_scores.topk(k=topk, dim=1, largest=True)
    avg_scores = topk_scores.values.mean(dim=1).reshape(1, 1, shape_s[0], shape_s[1])
    scores = F.interpolate(avg_scores, shape_l, mode='bilinear', align_corners=False)

    return scores

if __name__ == '__main__':

    # Hyper parameters
    parser = argparse.ArgumentParser(description='Feature Templates')

    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to feature extractor network checkpoint (default: none).')
    parser.add_argument(
        '-s', '--source', default=None, type=int,
        help='Source of feature templates (default: none, 0: dataset, 1: addon, 2: eval).')
    args = parser.parse_args()

    print('Args:', args)

    feature_templates = FeatureTemplates(args.checkpoint)

    if args.source == 0:
        feature_templates.build_from_dataset()
    elif args.source == 1:
        feature_templates.build_from_addon()
    elif args.source == 2:
        pass
        # feature_templates.build_from_eval()