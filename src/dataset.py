import os
import sys
import random
import copy
import numpy as np
from glob import glob
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data

from src.utils import load_image_in_PIL
import src.transforms as my_tf


class WaterDataset(data.Dataset):

    def __init__(self, mode, dataset_path, input_size=None, test_case=None, eval_size=None):

        super(WaterDataset, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.test_case = test_case
        self.img_list = []
        self.label_list = []
        self.verbose_flag = False
        self.online_augmentation_per_epoch = 640
        self.eval_size = eval_size

        if mode == 'train_offline':
            water_subdirs = ['ADE20K', 'river_segs']

            print('Initialize offline training dataset:')

            for sub_folder in water_subdirs:
                img_path = os.path.join(dataset_path, 'JPEGImages/', sub_folder)
                img_list = os.listdir(img_path)
                img_list.sort(key=lambda x: (len(x), x))
                self.img_list += [os.path.join(img_path, name) for name in img_list]

                label_path = os.path.join(dataset_path, 'Annotations/', sub_folder)
                label_list = os.listdir(label_path)
                label_list.sort(key=lambda x: (len(x), x))
                self.label_list += [os.path.join(label_path, name) for name in label_list]

                print('Add', sub_folder, len(img_list), 'files.')

        elif mode == 'train_online':
            if test_case is None:
                raise ('test_case can not be None.')

            img_path = os.path.join(dataset_path, 'JPEGImages/', test_case)
            img_list = os.listdir(img_path)
            img_list.sort(key=lambda x: (len(x), x))

            first_frame_path = os.path.join(dataset_path, 'JPEGImages/', test_case, img_list[0])
            first_frame_label_path = os.path.join(dataset_path, 'Annotations/', test_case, img_list[0])

            # Detect label image format: png or jpg
            first_frame_label_path = first_frame_label_path[:-3]
            if os.path.exists(first_frame_label_path + 'png'):
                first_frame_label_path += 'png'
            else:
                first_frame_label_path += 'jpg'

            # print(first_frame_path, first_frame_label_path)

            self.first_frame = load_image_in_PIL(first_frame_path).convert('RGB')
            self.first_frame_label = load_image_in_PIL(first_frame_label_path).convert('L')

            # print(self.first_frame)
            # print(self.first_frame_label)
            # x = self.first_frame.copy()
            # self.first_frame.save('tmp/first_frame_-1.png')

        elif mode == 'eval':
            if test_case is None:
                raise ('test_case can not be None.')

            img_path = os.path.join(dataset_path, 'JPEGImages/', test_case)
            img_list = os.listdir(img_path)
            img_list.sort(key=lambda x: (len(x), x))
            self.img_list = [os.path.join(img_path, name) for name in img_list]

            first_frame_label_path = os.path.join(dataset_path, 'Annotations/', test_case, img_list[0])

            # Detect label image format: png or jpg
            first_frame_label_path = first_frame_label_path[:-3]
            if os.path.exists(first_frame_label_path + 'png'):
                first_frame_label_path += 'png'
            else:
                first_frame_label_path += 'jpg'

            if not os.path.exists(first_frame_label_path):
                label_list = glob(os.path.join(dataset_path, 'Annotations/', test_case, '*.png'))
                label_list.sort(key=lambda x: (x, len(x)))
                first_frame_label_path = label_list[0]

            self.first_frame = load_image_in_PIL(self.img_list[0]).convert('RGB')
            self.img_list.pop(0)

            self.first_frame_label = load_image_in_PIL(first_frame_label_path).convert('L')

            if self.eval_size:
                self.origin_size = self.first_frame.size
                self.first_frame = self.first_frame.resize(self.eval_size, Image.ANTIALIAS)
                self.first_frame_label = self.first_frame_label.resize(self.eval_size, Image.ANTIALIAS)

        else:
            raise ('Mode %s does not support in [train_offline, train_online, eval].' % mode)

    def __len__(self):
        if self.mode == 'train_online':
            return self.online_augmentation_per_epoch
        else:
            return len(self.img_list)

    def get_first_frame(self):
        img_tf = TF.to_tensor(self.first_frame)
        img_tf = my_tf.imagenet_normalization(img_tf)
        return img_tf

    def get_first_frame_label(self):
        return TF.to_tensor(self.first_frame_label)

    def __getitem__(self, index):
        raise NotImplementedError


class WaterDataset_RGB(WaterDataset):

    def __init__(self, mode, dataset_path, input_size=None, test_case=None, eval_size=None):

        super(WaterDataset_RGB, self).__init__(mode, dataset_path, input_size, test_case, eval_size)

    def __getitem__(self, index):

        if self.mode == 'train_offline':
            img = load_image_in_PIL(self.img_list[index]).convert('RGB')
            label = load_image_in_PIL(self.label_list[index]).convert('L')

            sample = self.apply_transforms(img, label)

        elif self.mode == 'train_online':
            sample = self.apply_transforms(self.first_frame, self.first_frame_label)

        elif self.mode == 'eval':
            img = load_image_in_PIL(self.img_list[index]).convert('RGB')
            if self.eval_size:
                img = img.resize(self.eval_size, Image.ANTIALIAS)
            sample = self.apply_transforms(img)

        return sample

    def resize_to_origin(self, img):
        return img.resize(self.origin_size)

    def apply_transforms(self, img, label=None):

        if self.mode == 'train_offline' or self.mode == 'train_online':

            img = my_tf.random_adjust_color(img, self.verbose_flag)
            img, label = my_tf.random_affine_transformation(img, None, label, self.verbose_flag)
            img, label = my_tf.random_resized_crop(img, None, label, self.input_size, self.verbose_flag)

        elif self.mode == 'eval':
            pass

        img = TF.to_tensor(img)
        img = my_tf.imagenet_normalization(img)

        if self.mode == 'train_offline' or self.mode == 'train_online':

            label = TF.to_tensor(label)

            # if (img.shape[0] != 3):
            # print('img', img.shape)
            # print('mask', mask.shape)
            # print('label', label.shape)

            sample = {
                'img': img,
                'label': label
            }
        else:
            sample = {
                'img': img
            }

        return sample
