import os
import sys
import numpy as np
import cv2
from torch.utils import data

class WaterDataset(data.Dataset):

    def __init__(self, 
                 mode,
                 dataset_path, 
                 img_transforms=None, 
                 label_transforms=None):
        
        self.mode = mode
        if mode != 'test':
            self.label_list = []
            self.label_transforms = label_transforms
            water_subdirs = ['ADE20K', 'buffalo0', 'canal0', 'creek0', 'lab0', 'stream0', 'stream1', 'stream2']

        self.img_list = []
        self.img_transforms = img_transforms
        
        if mode != 'test':
            
            for sub_folder in water_subdirs:
                
                imgs_path = os.path.join(dataset_path, 'imgs/', sub_folder)
                imgs_list = os.listdir(imgs_path)
                imgs_list.sort(key = lambda x: (len(x), x))
                self.img_list += [os.path.join(imgs_path, name) for name in imgs_list]

                labels_path = os.path.join(dataset_path, 'labels/', sub_folder)
                labels_list = os.listdir(labels_path)
                labels_list.sort(key = lambda x: (len(x), x))
                self.label_list += [os.path.join(labels_path, name) for name in labels_list]

        else:
            imgs_list = os.listdir(dataset_path)
            imgs_list.sort(key = lambda x: (len(x), x))
            self.img_list += [os.path.join(dataset_path, name) for name in imgs_list]

    def __getitem__(self, index):
        
        img = cv2.imread(self.img_list[index])
        
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        if self.mode != 'test':
            label = cv2.imread(self.label_list[index], 0)
            label = np.expand_dims(label, 2)

            if self.label_transforms is not None:
                label = self.label_transforms(label)
            
            return img, label

        return img


    def __len__(self):
        return len(self.img_list)