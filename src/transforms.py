import random
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomResizedCrop


def random_adjust_color(img):
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.2, 1.1)
        img = TF.adjust_brightness(img, brightness_factor)

    if random.random() < 0.5:
        contrast_factor = random.uniform(0.5, 1.5)
        img = TF.adjust_contrast(img, contrast_factor)

    if random.random() < 0.5:
        hue_factor = random.uniform(-0.05, 0.05)
        img = TF.adjust_hue(img, hue_factor)

    return img

def random_affine_transformation(img, mask, label):
    
    if random.random() < 0.5:
        degrees = 30
        translate = (0.2, 0.2)
        scale = (0.7, 1.3)
        shear = 0.3
        resample = Image.BICUBIC

        img = TF.affine(img, degrees, translate, scale, shear, resample)
        label = TF.affine(img, degrees, translate, scale, shear, resample)
    
    if random.random() < 0.5:
        
        img = TF.vflip(img)
        mask = TF.vflip(mask)
        label = TF.vflip(label)

    return img, mask, label

def random_mask_perturbation(mask):

    degrees = 20
    translate = (0.1, 0.1)
    scale = (0.8, 1.2)
    shear = 0.2
    resample = Image.BICUBIC

    mask = TF.affine(mask, degrees, translate, scale, shear, resample)

    morphologic_times = int(random.random() * 10)
    mask /= 255

    for i in range(morphologic_times):
        if random.random() < 0.5:
            mask = binary_dilation(mask)
        else:
            mask = binary_erosion(mask)

    mask = np.uint8(mask) * 255

    return mask

def random_resized_crop(img, mask, label, size):
    
    return img, mask, label

def imagenet_normalization(img_tensor):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img_tensor = TF.normalize(img_tensor, mean, std)

    return img_tensor


training_size = (224, 224, 3)
random_resized_crop = RandomResizedCrop(training_size, scale=(0,5, 1.5), interpolation=Image.BICUBIC)