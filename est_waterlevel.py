import os
import argparse
import configparser
import math
import copy
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.dates as mdates
from datetime import datetime
import cv2
import argparse

seg_thres = 128

def track_object(img_folder, overlay_folder, out_folder):

    tracker = cv2.TrackerKCF_create()

    img_list = os.listdir(img_folder)
    img_list.sort(key = lambda x: (len(x), x))

    overlays_list = os.listdir(overlay_folder)
    overlays_list.sort(key = lambda x: (len(x), x))

    # First frame
    frame_st = cv2.imread(os.path.join(img_folder, img_list[0]))
    overlays_st = cv2.imread(os.path.join(overlay_folder, overlays_list[0]))
    print(frame_st.shape)
    print(overlays_st.shape)
    frame_st = cv2.resize(frame_st, (overlays_st.shape[1], overlays_st.shape[0]))
    
    bbox_st = (768, 223, 46, 42)
    while True:    
        bbox_st = cv2.selectROI('First Frame', frame_st, fromCenter=False)
        if bbox_st[2] > 0 and bbox_st[3] > 0:
            break
    x, y, w, h = [int(v) for v in bbox_st]
    cv2.rectangle(overlays_st, (x, y), (x+w, y + h), (0, 200, 0), 2)
    out_path = os.path.join(out_folder, img_list[0][:-4] + '_tag.png')
    cv2.imwrite(out_path, overlays_st)

    tracker.init(frame_st, bbox_st)
    bbox_old = bbox_st
    
    print('Init', bbox_st)

    key_pts = [(int(x + w/2), int(y + h))]

    for i in range(1, len(img_list)):

        img = cv2.imread(os.path.join(img_folder, img_list[i]))
        overlay = cv2.imread(os.path.join(overlay_folder, overlays_list[i]))
        img = cv2.resize(img, (overlays_st.shape[1], overlays_st.shape[0]))

        op_flag, bbox = tracker.update(img)

        print(op_flag, bbox)
        if op_flag:
            x, y, w, h = [int(v) for v in bbox]
            bbox_old = bbox
        else:
            x, y, w, h = [int(v) for v in bbox_old]
        cv2.rectangle(overlay, (x, y), (x+w, y + h), (0, 200, 0), 2)

        out_path = os.path.join(out_folder, img_list[i][:-4] + '_tag.png')
        cv2.imwrite(out_path, overlay)

        # cv2.imshow('overlay', overlay)
        # cv2.waitKey()

        key_pts.append((int(x + w/2), int(y + h)))
    
    return key_pts
        

def est_water_boundary(seg_folder, overlay_folder, out_folder, key_pts):

    seg_list = os.listdir(seg_folder)
    seg_list.sort(key = lambda x: (len(x), x))

    overlays_list = os.listdir(overlay_folder)
    overlays_list.sort(key = lambda x: (len(x), x))

    annotation = cv2.imread(os.path.join(seg_folder, seg_list[0]))
    annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2GRAY)
    annotation = cv2.GaussianBlur(annotation, (3, 3), 0)

    seg_size = annotation.shape[:2]
    grad_x = cv2.Sobel(annotation, cv2.CV_16S, 1, 0, None, 3)
    grad_y = cv2.Sobel(annotation, cv2.CV_16S, 0, 1, None, 3)

    overlay_st = cv2.imread(os.path.join(overlay_folder, overlays_list[0]))
    out_path = os.path.join(out_folder, overlays_list[0])    

    for y in range(key_pts[0][1] + 1, seg_size[0]):
        if annotation[y][key_pts[0][0]] > seg_thres:
            a = int(grad_x[y][key_pts[0][0]])
            b = int(grad_y[y][key_pts[0][0]])
            c = (a * a + b * b) ** 0.5
            assert(c > 0)
            sin_v = abs(b / c)

            cv2.line(overlay_st, key_pts[0], (key_pts[0][0], y), (200, 0, 0), 2)
            cv2.imwrite(out_path, overlay_st)

            break
    
    water_boundary_pts = copy.deepcopy(key_pts)

    for i in range(len(key_pts)):
        
        seg_path = os.path.join(seg_folder, seg_list[i])
        seg = cv2.imread(seg_path)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)

        out_path = os.path.join(out_folder, overlays_list[i])    
        
        for y in range(key_pts[i][1] + 1, seg_size[0]):
            if seg[y][key_pts[i][0]] > seg_thres:
                water_boundary_pts[i] = (key_pts[i][0], y)
                
                overlay = cv2.imread(os.path.join(overlay_folder, overlays_list[i]))
                cv2.line(overlay, key_pts[i], (key_pts[i][0], y), (200, 0, 0), 2)
                cv2.imwrite(out_path, overlay)
                # cv2.imshow('overlay', overlay)
                # cv2.waitKey()

                break
    
    return sin_v, water_boundary_pts


def get_time_arr(img_folder):

    img_list = os.listdir(img_folder)
    img_list.sort(key = lambda x: (len(x), x))

    time_arr = []

    for img_name in img_list:
        timestamp = img_name[-12:-4].replace('-', ':')
        timestamp = datetime.strptime(timestamp, '%H:%M:%S')
        time_arr.append(timestamp)
    
    return time_arr


def est_waterlevel(video_name, model_name='AANet'):

    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    img_folder = os.path.join(cfg['paths']['dataset_ubuntu'], 'test_videos', video_name)
    overlay_folder = os.path.join(cfg['paths']['dataset_ubuntu'], 'results', model_name + '_overlays', video_name)
    out_folder = os.path.join(cfg['paths']['dataset_ubuntu'], 'results', model_name + '_overlays_bbox', video_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    key_pts = track_object(img_folder, overlay_folder, out_folder)

    seg_folder = os.path.join(cfg['paths']['dataset_ubuntu'], 'results', model_name + '_segs', video_name)
    cos_v, water_boundary_pts = est_water_boundary(seg_folder, out_folder, out_folder, key_pts)

    key_pts_y = np.array(key_pts)[:, 1]
    water_boundary_pts_y = np.array(water_boundary_pts)[:, 1]

    print(len(key_pts_y), len(water_boundary_pts_y))

    gap = (water_boundary_pts_y - key_pts_y) * cos_v

    time_arr = get_time_arr(img_folder)
    time_fmt = mdates.DateFormatter('%H:%M')
    
    plt.plot(time_arr, gap)
    plt.gca().xaxis.set_major_formatter(time_fmt)
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Estimate water level.')
    parser.add_argument(
        '--video-name', type=str, required=True,
        help='Video name.')
    args = parser.parse_args()

    print('Args:', args)
    video_name = args.video_name

    # video_name = 'boston_harbor_20190119'

    est_waterlevel(video_name)

    