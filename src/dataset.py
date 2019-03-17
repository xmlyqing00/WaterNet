import os
import sys
import cv2
from torch.utils import data

class WaterDataset(data.Dataset):

    def __init__(self, 
                 mode,
                 dataset_path, 
                 input_transforms=None, 
                 target_transforms=None):
        
        self.mode = mode
        if mode != 'test':
            self.target_list = []
            self.target_transforms = target_transforms
            water_subdirs = ['ADE20K', 'buffalo0', 'canal0', 'creek0', 'lab0', 'stream0', 'stream1', 'stream2']

        self.input_list = []
        self.input_transforms = input_transforms
        
        if mode != 'test':
            
            for sub_folder in water_subdirs:
                
                imgs_path = os.path.join(dataset_path, 'imgs/', sub_folder)
                imgs_list = os.listdir(imgs_path)
                imgs_list.sort(key = lambda x: (len(x), x))
                self.input_list += [os.path.join(imgs_path, name) for name in imgs_list]

                labels_path = os.path.join(dataset_path, 'labels/', sub_folder)
                labels_list = os.listdir(labels_path)
                labels_list.sort(key = lambda x: (len(x), x))
                self.target_list += [os.path.join(labels_path, name) for name in labels_list]

        else:

            imgs_list = os.listdir(dataset_path)
            imgs_list.sort(key = lambda x: (len(x), x))
            self.input_list += [os.path.join(dataset_path, name) for name in imgs_list]

    def __getitem__(self, index):
        
        input = cv2.imread(self.input_list[index])

        if self.input_transforms is not None:
            input = self.input_transforms(input)

        if self.mode != 'test':
            target = cv2.imread(self.target_list[index], 0)

            if self.target_transforms is not None:
                target = self.target_transforms(target)
            
            return input, target

        return input


    def __len__(self):
        return len(self.input_list)