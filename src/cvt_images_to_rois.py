import cv2
import numpy as np
import os

def cvt_images_to_rois(image_folder, out_folder):

    image_list = os.listdir(image_folder)
    if (len(image_list) == 0):
        exit(-1)
    image_list.sort(key = lambda x: (len(x), x))

    x, y, w, h = 370, 80, 500, 500

    for image_name in image_list:

        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)

        out_path = os.path.join(out_folder, image_name)
        cv2.imwrite(out_path, image[y:y+h, x:x+w,:])
        
        print("Write", image_path)


if __name__ == '__main__':
    
    root_folder = '/Ship01/Dataset/water/collection/test_videos'
    test_name = 'boston_harbor2_small'
    out_name = test_name + '_rois'
    image_folder = os.path.join(root_folder, test_name)
    out_folder = os.path.join(root_folder, out_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    cvt_images_to_rois(image_folder, out_folder)