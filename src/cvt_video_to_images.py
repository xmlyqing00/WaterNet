import os
import numpy as np
import cv2

def cvt_video_series_to_images(video_series_path, video_series_name, out_frames_folder):
    
    if not os.path.exists(out_frames_folder):
        os.makedirs(out_frames_folder)

    video_list = os.listdir(video_series_path)
    video_list.sort(key = lambda x: (len(x), x))
    stride = 30
    frames_n = 100
    frames_cnt = 0

    for video_name in video_list:

        video_path = os.path.join(video_series_path, video_name)
        video = cv2.VideoCapture(video_path)
        cnt = 0

        print('\tVideo path:', video_path)
        print('\t\t')
        
        while (video.isOpened()):

            ret, frame = video.read()
            if not ret:
                break

            if cnt % stride == 0:
                print(cnt, end=' ')
                out_path = os.path.join(out_frames_folder, video_name[:-4] + '_%d.png' % cnt)
                cv2.imwrite(out_path, frame)
            
            cnt += 1
            frames_cnt += 1
            if frames_cnt > frames_n:
                break
            
        print('')
        # break


if __name__ == '__main__':

    root_folder = '/Ship01/Dataset/water/beaches/'
    out_folder = '/Ship01/Dataset/water/collection/test_videos/'
    video_series_list = os.listdir(root_folder)
    
    for video_series_name in video_series_list:

        video_series_path = os.path.join(root_folder, video_series_name)
        out_frames_folder = os.path.join(out_folder, video_series_name)

        print('Video series path:', video_series_name)
        print('Out frames folder:', out_frames_folder)
        cvt_video_series_to_images(video_series_path, video_series_name, out_frames_folder)
