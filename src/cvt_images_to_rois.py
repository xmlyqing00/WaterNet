import cv2
import numpy as np
import os

def cvt_images_to_rois(image_folder,
                        video_path,
                        fps = 30,
                        fourcc = cv2.VideoWriter_fourcc(*'XVID'),
                        stride=1):

    image_list = os.listdir(image_folder)
    if (len(image_list) == 0):
        exit(-1)
    image_list.sort(key = lambda x: (len(x), x))

    first_image_path = os.path.join(image_folder, image_list[0])
    height, width, channels = cv2.imread(first_image_path).shape
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    stride = max(0, int(stride))
    for image_idx in range(0, len(image_list), stride):

        image_path = os.path.join(image_folder, image_list[image_idx])
        image = cv2.imread(image_path)
        video.write(image)
        
        print("Write", image_path)

    video.release()
    

if __name__ == '__main__':
    
    root_folder = '/Ship01/Dataset/water/collection/test_videos'
    test_name = 'boston_harbor2_small'
    out_name = test_name + '_rois'
    image_folder = os.path.join(root_folder, test_name)
    out_folder = os.path.join(root_folder, out_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    cvt_images_to_rois(image_folder, out_folder)
    cvt_images_to_video(image_folder, video_path, fps, fourcc, stride)