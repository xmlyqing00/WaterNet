import os
import sys
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data

import src.transforms as my_tf


class WaterDataset(data.Dataset):

    def __init__(self, mode, dataset_path, input_size=None, test_case=None):
        
        self.mode = mode
        self.input_size = input_size
        self.test_case = test_case
        self.img_list = []
        self.label_list = []
        self.verbose_flag = False
        
        if mode == 'train_offline':
            water_subdirs = ['ADE20K', 'buffalo0', 'canal0', 'creek0', 'lab0', 'stream0', 'stream1', 'stream2']

            for sub_folder in water_subdirs:
                img_path = os.path.join(dataset_path, 'imgs/', sub_folder)
                img_list = os.listdir(img_path)
                img_list.sort(key = lambda x: (len(x), x))
                self.img_list += [os.path.join(img_path, name) for name in img_list]

                label_path = os.path.join(dataset_path, 'labels/', sub_folder)
                label_list = os.listdir(label_path)
                label_list.sort(key = lambda x: (len(x), x))
                self.label_list += [os.path.join(label_path, name) for name in label_list]

        elif mode == 'train_online':
            if test_case is None:
                raise('test_case can not be None.')

            label_path = os.path.join(dataset_path, 'labels/', test_case)
            label_list = os.listdir(label_path)
            label_list.sort(key = lambda x: (len(x), x))

            first_frame_label_path = os.path.join(dataset_path, 'labels/', test_case, label_list[0])
            first_frame_path = os.path.join(dataset_path, 'imgs/', test_case, label_list[0])

            self.first_frame_label = Image.open(first_frame_label_path)
            self.first_frame = Image.open(first_frame_path)

        elif mode == 'eval':
            if test_case is None:
                raise('test_case can not be None.')
            
            img_path = os.path.join(dataset_path, 'imgs/', test_case)
            img_list = os.listdir(img_path)
            img_list.sort(key = lambda x: (len(x), x))
            self.img_list = [os.path.join(img_path, name) for name in img_list]

            first_frame_label_path = os.path.join(dataset_path, 'labels/', test_case, img_list[0])
            self.img_list.pop(0)
            self.first_frame_label = Image.open(first_frame_label_path)

        else:
            raise('Mode %s does not support in [train_offline, train_online, eval].' % mode)

    def __getitem__(self, index):
        
        if self.mode == 'train_offline':
            
            img = Image.open(self.img_list[index])
            label = Image.open(self.label_list[index]).convert('L')
            
            sample = self.apply_transforms(img, label, label)
            return sample

        elif self.mode == 'train_online':
            sample = self.apply_transforms(self.first_frame, self.first_frame_label, self.first_frame_label)
            return sample

        elif self.mode == 'eval':
            img = Image.open(self.img_list[index])
            sample = self.apply_transforms(img)
            return sample
    
    def __len__(self):
        return len(self.img_list)

    def get_first_frame_label(self):
        return self.first_frame_label

    def apply_transforms(self, img, mask=None, label=None):

        if self.mode == 'train_offline':
            
            # img.save('tmp/ori_img.png')
            # mask.save('tmp/ori_mask.png')
            # label.save('tmp/ori_label.png')
            
            img = my_tf.random_adjust_color(img, self.verbose_flag)
            # img.save('tmp/color_img.png')

            img, mask, label = my_tf.random_affine_transformation(img, mask, label, self.verbose_flag)

            # img.save('tmp/affine_img.png')
            # mask.save('tmp/affine_mask.png')
            # label.save('tmp/affine_label.png')

            mask = my_tf.random_mask_perturbation(mask, self.verbose_flag)

            # mask.save('tmp/pertur_mask.png')
            img, mask, label = my_tf.random_resized_crop(img, mask, label, self.input_size, self.verbose_flag)

            # img.save('tmp/crop_img.png')
            # mask.save('tmp/crop_mask.png')
            # label.save('tmp/crop_label.png')

            mask = TF.to_tensor(mask)
            label = TF.to_tensor(label)


        elif self.mode == 'train_online':
            
            img = my_tf.random_adjust_color(img)
            img, mask, label = my_tf.random_affine_transformation(img, mask, label)
            mask = my_tf.random_mask_perturbation(mask)
            img, mask, label = my_tf.random_resized_crop(img, mask, label, self.input_size)

            mask = TF.to_tensor(mask)
            label = TF.to_tensor(label)

        elif self.mode == 'eval':
            pass

        img = TF.to_tensor(img)
        img = my_tf.imagenet_normalization(img)

        sample = {
            'img': img,
            'mask': mask,
            'label': label
        }

        return sample

    