from PIL import Image
import torch

eps = 1e-8

def load_image_in_PIL(path):
    img = Image.open(path)
    img.load() # Very important for loading large image
    return img

def iou_tensor(output: torch.Tensor, label: torch.Tensor):

    intersection = (output & label).float().sum((1, 2))  
    union = (output | label).float().sum((1, 2))         
    iou = (intersection + eps) / (union + eps)  
    
    return iou