import os
import sys
import random
import copy
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data

from src.utils import load_image_in_PIL
import src.transforms as my_tf


class WaterDataset(data.Dataset):

    def __init__(self, mode, dataset_path, input_size=None, test_case=None):
        
        self.mode = mode
        self.input_size = input_size
        self.test_case = test_case
        self.img_list = []
        self.label_list = []
        self.verbose_flag = False
        self.online_augmentation_per_epoch = 6400
        
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

            img_path = os.path.join(dataset_path, 'test_videos/', test_case)
            img_list = os.listdir(img_path)
            img_list.sort(key = lambda x: (len(x), x))

            first_frame_path = os.path.join(dataset_path, 'test_videos/', test_case, img_list[0])
            first_frame_label_path = os.path.join(dataset_path, 'test_annots/', test_case, img_list[0])

            # Detect label image format: png or jpg
            first_frame_label_path = first_frame_label_path[:-3]
            if os.path.exists(first_frame_label_path + 'png'):
                first_frame_label_path += 'png'
            else:
                first_frame_label_path += 'jpg'

            # print(first_frame_path, first_frame_label_path)

            self.first_frame = load_image_in_PIL(first_frame_path)
            self.first_frame_label = load_image_in_PIL(first_frame_label_path).convert('L')

            # print(self.first_frame)
            # print(self.first_frame_label)
            # x = self.first_frame.copy()
            # self.first_frame.save('tmp/first_frame_-1.png')

        elif mode == 'eval':
            if test_case is None:
                raise('test_case can not be None.')
            
            img_path = os.path.join(dataset_path, 'test_videos/', test_case)
            img_list = os.listdir(img_path)
            img_list.sort(key = lambda x: (len(x), x))
            self.img_list = [os.path.join(img_path, name) for name in img_list]

            first_frame_label_path = os.path.join(dataset_path, 'test_annots/', test_case, img_list[0])

            # Detect label image format: png or jpg
            first_frame_label_path = first_frame_label_path[:-3]
            if os.path.exists(first_frame_label_path + 'png'):
                first_frame_label_path += 'png'
            else:
                first_frame_label_path += 'jpg'

            self.img_list.pop(0)
            self.first_frame_label = load_image_in_PIL(first_frame_label_path).convert('L')

        else:
            raise('Mode %s does not support in [train_offline, train_online, eval].' % mode)

    def __getitem__(self, index):
        
        if self.mode == 'train_offline':
            img = load_image_in_PIL(self.img_list[index])
            label = load_image_in_PIL(self.label_list[index]).convert('L')
            
            sample = self.apply_transforms(img, label, label)
            return sample

        elif self.mode == 'train_online':
            sample = self.apply_transforms(self.first_frame, self.first_frame_label, self.first_frame_label)
            return sample

        elif self.mode == 'eval':
            img = load_image_in_PIL(self.img_list[index])
            sample = self.apply_transforms(img)
            return sample
    
    def __len__(self):
        if self.mode == 'train_online':
            return self.online_augmentation_per_epoch
        else:
            return len(self.img_list)

    def get_first_frame_label(self):
        return TF.to_tensor(self.first_frame_label)

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

        elif self.mode == 'train_online':
            
            # img.save('tmp/img_ori.png')

            img = my_tf.random_adjust_color(img, self.verbose_flag)
            img, mask, label = my_tf.random_affine_transformation(img, mask, label, self.verbose_flag)
            mask = my_tf.random_mask_perturbation(mask, self.verbose_flag)
            img, mask, label = my_tf.random_resized_crop(img, mask, label, self.input_size, self.verbose_flag)

            # img.save('tmp/img_crop.png')
            # mask.save('tmp/mask_crop.png')
            # label.save('tmp/label_crop.png')

        elif self.mode == 'eval':
            pass

        img = TF.to_tensor(img)
        img = my_tf.imagenet_normalization(img)

        if self.mode == 'train_offline' or self.mode == 'train_online':

            mask = TF.to_tensor(mask)
            label = TF.to_tensor(label)

            sample = {
                'img': img,
                'mask': mask,
                'label': label
            }
        else:
            sample = {
                'img': img
            }

        return sample
