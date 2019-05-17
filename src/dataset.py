import os
import sys
import random
import copy
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data

from src.utils import load_image_in_PIL
import src.transforms as my_tf


class WaterDataset(data.Dataset):

    def __init__(self, mode, dataset_path, input_size=None, test_case=None):
        
        super(WaterDataset, self).__init__()

        self.mode = mode
        self.input_size = input_size
        self.test_case = test_case
        self.img_list = []
        self.label_list = []
        self.verbose_flag = False
        self.online_augmentation_per_epoch = 640
        
        if mode == 'train_offline':
            # water_subdirs = ['ADE20K', 'buffalo0', 'canal0', 'creek0', 'lab0', 'stream0', 'stream1', 'stream2', 'river_segs']
            water_subdirs = ['ADE20K', 'river_segs']

            print('Initialize offline training dataset:')

            for sub_folder in water_subdirs:
                img_path = os.path.join(dataset_path, 'imgs/', sub_folder)
                img_list = os.listdir(img_path)
                img_list.sort(key = lambda x: (len(x), x))
                self.img_list += [os.path.join(img_path, name) for name in img_list]

                label_path = os.path.join(dataset_path, 'labels/', sub_folder)
                label_list = os.listdir(label_path)
                label_list.sort(key = lambda x: (len(x), x))
                self.label_list += [os.path.join(label_path, name) for name in label_list]

                print('Add', sub_folder, len(img_list), 'files.')

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

            self.first_frame = load_image_in_PIL(first_frame_path).convert('RGB')
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

    def __len__(self):
        if self.mode == 'train_online':
            return self.online_augmentation_per_epoch
        else:
            return len(self.img_list)

    def get_first_frame_label(self):
        return TF.to_tensor(self.first_frame_label)

    def __getitem__(self, index):
        raise NotImplementedError


class WaterDataset_OSVOS(WaterDataset):

    def __init__(self, mode, dataset_path, input_size=None, test_case=None):

        super(WaterDataset_OSVOS, self).__init__(mode, dataset_path, input_size, test_case)
        self.eval_size = (640, 640)

    def __getitem__(self, index):
        
        if self.mode == 'train_offline':
            img = load_image_in_PIL(self.img_list[index]).convert('RGB')
            label = load_image_in_PIL(self.label_list[index]).convert('L')

            sample = self.apply_transforms(img, label)

        elif self.mode == 'train_online':
            sample = self.apply_transforms(self.first_frame, self.first_frame_label)

        elif self.mode == 'eval':
            img = load_image_in_PIL(self.img_list[index]).convert('RGB')
            self.origin_size = img.size
            img.thumbnail(self.eval_size, Image.ANTIALIAS)
            # print(self.origin_size, img.size)
            sample = self.apply_transforms(img)
        
        return sample

    def resize_to_origin(self, img):
        return img.resize(self.first_frame_label.size)

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

class WaterDataset_PFD(WaterDataset):

    def __init__(self, mode, dataset_path, input_size=None, test_case=None):

        super(WaterDataset_PFD, self).__init__(mode, dataset_path, input_size, test_case)
        self.class_n = 2
        self.label_colors = [255, 0]

        self.black_tensor1 = TF.to_tensor(np.zeros(input_size + [1], np.uint8))
        self.black_tensor3 = TF.to_tensor(np.zeros(input_size + [3], np.uint8))

    def __getitem__(self, index):

        if self.mode == 'train_offline':
            img = load_image_in_PIL(self.img_list[index]).convert('RGB')
            label = load_image_in_PIL(self.label_list[index]).convert('L')

            sample = self.apply_transforms(img, label)
            return sample

        elif self.mode == 'train_online':
            sample = self.apply_transforms(self.first_frame, self.first_frame_label)
            return sample

        elif self.mode == 'eval':
            img = load_image_in_PIL(self.img_list[index]).convert('RGB')
            sample = self.apply_transforms(img)
            return sample

    def apply_transforms(self, img, label=None):
        
        if self.mode == 'train_offline' or self.mode == 'train_online':
            
            # img.save('tmp/ori_img.png')
            # label.save('tmp/ori_label.png')
            
            img = my_tf.random_adjust_color(img, self.verbose_flag)
            img, label = my_tf.random_affine_transformation(img, None, label, self.verbose_flag)

            # img.save('tmp/affine_img.png')
            # label.save('tmp/affine_label.png')

            img, label = my_tf.random_resized_crop(img, None, label, self.input_size, self.verbose_flag)

            # img.save('tmp/crop_img.png')
            # label.save('tmp/crop_label.png')

        elif self.mode == 'eval':
            pass

        img = TF.to_tensor(img)
        img = my_tf.imagenet_normalization(img)

        if self.mode == 'train_offline' or self.mode == 'train_online':

            label = TF.to_tensor(label)

            label_water = label == 1
            mask = torch.cat((label_water, label_water, label_water), 0)
            img_water = torch.where(mask, img, self.black_tensor3)
            label_water = torch.cat((label_water.float(), self.black_tensor1), 0)

            label_bg = ~mask
            img_bg = torch.where(label_bg, img, self.black_tensor3)
            label_bg = label_bg[0].unsqueeze(0)
            label_bg = torch.cat((self.black_tensor1, label_bg.float()), 0)
        
            sample = {
                'img': torch.cat((img_water, img_bg), 0),
                'label': torch.cat((label_water, label_bg), 0)
            }
        else:
            sample = {
                'img': img
            }

        return sample


class WaterDataset_RGBMask(WaterDataset):

    def __init__(self, mode, dataset_path, input_size=None, test_case=None):

        super(WaterDataset_RGBMask, self).__init__(mode, dataset_path, input_size, test_case)

    def __getitem__(self, index):
        
        if self.mode == 'train_offline':
            img = load_image_in_PIL(self.img_list[index]).convert('RGB')
            label = load_image_in_PIL(self.label_list[index]).convert('L')
            
            sample = self.apply_transforms(img, label, label)
            return sample

        elif self.mode == 'train_online':
            sample = self.apply_transforms(self.first_frame, self.first_frame_label, self.first_frame_label)
            return sample

        elif self.mode == 'eval':
            img = load_image_in_PIL(self.img_list[index]).convert('RGB')
            sample = self.apply_transforms(img)
            return sample

    def apply_transforms(self, img, mask=None, label=None):

        if self.mode == 'train_offline' or self.mode == 'train_online':
            
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

        elif self.mode == 'eval':
            pass

        img = TF.to_tensor(img)
        img = my_tf.imagenet_normalization(img)

        if self.mode == 'train_offline' or self.mode == 'train_online':

            mask = TF.to_tensor(mask)
            label = TF.to_tensor(label)

            # if (img.shape[0] != 3):
                # print('img', img.shape)
            # print('mask', mask.shape)
            # print('label', label.shape)
            
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