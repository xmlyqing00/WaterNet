import cv2
import numpy as np
import os
from tqdm import tqdm

def cvt_images_to_video(image_folder,
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
    for image_idx in tqdm(range(0, len(image_list), stride)):

        image_path = os.path.join(image_folder, image_list[image_idx])
        image = cv2.imread(image_path)
        video.write(image)
        
        # print("Write", image_path)

    video.release()

    print(f'Save to {video_path}')
    

if __name__ == '__main__':
    
    root_folder = '/Ship01/Dataset/water/results/'
    method = 'WaterNet'
    video_name = 'boston_harbor_20190119_20190123_day'
    
    image_folder = os.path.join(root_folder, method + '_overlays_bbox/', video_name)
    video_folder = os.path.join(root_folder, method + '_videos/')
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    video_path = os.path.join(video_folder, video_name + '.mp4')
    fps = 10
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    stride = 1

    cvt_images_to_video(image_folder, video_path, fps, fourcc, stride)
