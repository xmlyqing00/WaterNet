import cv2
import numpy as np
import os

from .cvt_object_label import cvt_object_label

def add_mask_to_image(image, mask, label_color):
    
    # cv2.imshow("input_mask", mask)

    roi = cvt_object_label(mask, label_color, [255, 255, 255])
    mask = cvt_object_label(mask, label_color, [255, 0, 0])
    
    complement = cv2.bitwise_not(roi)
    complement_img = cv2.bitwise_and(complement, image)
    mask = mask + complement_img

    # cv2.imshow("roi", roi)
    # cv2.imshow("com", complement_img)
    # cv2.imshow("img", image)
    # cv2.imshow("mask", mask)
    # cv2.waitKey()

    alpha = 0.5
    image_mask = image.copy()
    cv2.addWeighted(image, alpha, mask, 1 - alpha, 0, image_mask)

    return image_mask

def binary_threshold(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def cvt_images_to_overlays(image_folder, 
                           mask_folder,
                           output_folder, 
                           label_color=[255, 255, 255],
                           stride=1,
                           frame_st=0,
                           eval_size=None):

    image_list = os.listdir(image_folder)
    mask_list = os.listdir(mask_folder)

    if (len(image_list) == 0):
        exit(-1)
    
    # assert(len(image_list) == len(mask_list))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_list.sort(key = lambda x: (len(x), x))
    mask_list.sort(key = lambda x: (len(x), x))

    stride = max(0, int(stride))
    for image_idx in range(0, len(image_list), stride):
        
        image_path = os.path.join(image_folder, image_list[image_idx+frame_st])
        image = cv2.imread(image_path)
        if eval_size:
            image = cv2.resize(image, (eval_size[1], eval_size[0]), None)

        mask_path = os.path.join(mask_folder, mask_list[image_idx])
        mask = cv2.imread(mask_path)
        mask = binary_threshold(mask)

        origin_shape = image.shape
        if mask.shape != image.shape:
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), None)

        image_mask = add_mask_to_image(image, mask, label_color)

        if image_mask.shape != origin_shape:
            image_mask = cv2.resize(image_mask, (origin_shape[1], origin_shape[0]), None)

        filename, ext = os.path.splitext(image_list[image_idx])
        output_name = filename + '_mask.png'
        output_path = os.path.join(output_folder, output_name)
        cv2.imwrite(output_path, image_mask)

        # print(f'Add mask to image: {image_idx} {output_path}', end='\r')

    # print('')
    print('Add masks to images. Overlays folder:', output_folder)


def run_cvt_images_to_overlays(video_name, mask_folder, root_folder, model_name='RGBMaskNet', eval_size=None):

    image_folder = os.path.join(root_folder, 'test_videos/', video_name)
    mask_folder = os.path.join(root_folder, 'results', model_name + '_segs', mask_folder)
    output_folder = os.path.join(root_folder, 'results', model_name + '_overlays', video_name)
    label_color = (255, 255, 255)
    stride = 1
    frame_st = 0

    # eval_size: (h, w)
    cvt_images_to_overlays(image_folder, mask_folder, output_folder, label_color, stride, frame_st, eval_size)

def run_add_mask_to_image():

    root_folder = '/Ship01/Documents/MyPapers/FloodHydrograph/imgs'
    image = cv2.imread(os.path.join(root_folder, '1798_original.png'))
    mask = cv2.imread(os.path.join(root_folder, '1798_before_smoothed.png'))
    image_mask = add_mask_to_image(image, mask, [200, 0, 0])
    cv2.imwrite(os.path.join(root_folder, '1798_before_smoothed_overlay.png'), image_mask)

if __name__ == '__main__':
    
    root_folder = '/Ship01/Dataset/water/'
    video_name_set = ['canal0', 'stream0', 'stream1', 'stream3_small', 'stream4', 'buffalo0_small', 'boston_harbor2_small_rois']
    model_name = 'RGMP'

    for video_folder in video_name_set:
        run_cvt_images_to_overlays(video_folder, video_folder, root_folder, model_name)
    # run_add_mask_to_image()