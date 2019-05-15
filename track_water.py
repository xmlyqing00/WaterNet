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

def track_water(video_name):

    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    img_folder = os.path.join(cfg['paths']['dataset'], 'test_videos', video_name)

    tracker = cv2.TrackerKCF_create()

    img_list = os.listdir(img_folder)
    img_list.sort(key = lambda x: (len(x), x))

    frame_st = cv2.imread(os.path.join(img_folder, img_list[0]))
    bbox_st = []
    while True:    
        bbox_st = cv2.selectROI('First Frame', frame_st, fromCenter=False)
        if bbox_st[2] > 0 and bbox_st[3] > 0:
            break
    x, y, w, h = [int(v) for v in bbox_st]
    cv2.rectangle(frame_st, (x, y), (x+w, y + h), (0, 200, 0), 2)

    tracker.init(frame_st, bbox_st)
    bbox_old = bbox_st

    print(bbox_old)
    cv2.imshow('frame', frame_st)
    cv2.waitKey()

    for i in range(1, len(img_list)):

        frame = cv2.imread(os.path.join(img_folder, img_list[i]))

        op_flag, bbox = tracker.update(frame)
        print(op_flag, bbox)

        if op_flag:
            x, y, w, h = [int(v) for v in bbox]
            bbox_old = bbox
        else:
            x, y, w, h = [int(v) for v in bbox_old]
        cv2.rectangle(frame, (x, y), (x+w, y + h), (0, 200, 0), 2)

        cv2.imshow('frame', frame)
        cv2.waitKey()


if __name__ == '__main__':

    video_name = 'stream1'

    track_water(video_name)