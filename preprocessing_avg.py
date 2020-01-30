import os
import argparse
import configparser
import math
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from datetime import datetime
import cv2
import argparse
from tqdm import tqdm
from pandas.plotting import register_matplotlib_converters
        
good_match_thres = 0.8
min_match_count_thres = 6

def select_rois(img, roi_num):

    img_size = img.shape[:2]
    mask = np.zeros(img_size, dtype=np.uint8)
    # cv2.imshow('mask', mask)

    for i in range(roi_num):
        while True:    
            roi_bbox = cv2.selectROI('Select ROIs', img, fromCenter=False)
            if roi_bbox[2] > 0 and roi_bbox[3] > 0:
                break
        
        x, y, w, h = [int(v) for v in roi_bbox]
        # print(roi_bbox)
        mask[y:y+h, x:x+w] = 255
        # cv2.imshow('mask', mask)
        # cv2.waitKey(1)

    return mask


def align_imgs(img_folder, mask_folder, out_img_folder, roi_num):

    img_list = os.listdir(img_folder)
    img_list.sort(key=lambda x: (len(x), x))

    mask_list = os.listdir(mask_folder)
    mask_list.sort(key=lambda x: (len(x), x))

    # First frame
    img_st = cv2.imread(os.path.join(img_folder, img_list[0]))
    mask = select_rois(img_st, roi_num)
    cv2.imwrite('tmp/mask.png', mask)
    surf = cv2.xfeatures2d.SURF_create(1000)

    kpt_st, des_st = surf.detectAndCompute(img_st, mask)
    # img_st_kpt = cv2.drawKeypoints(img_st, kpt_st, None, (255, 0, 0), 4)

    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for i in tqdm(range(1, len(img_list))):

        img_cur = cv2.imread(os.path.join(img_folder, img_list[i]))
        kpt_cur, des_cur = surf.detectAndCompute(img_cur, mask)

        matches = flann.knnMatch(des_st, des_cur, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < good_match_thres * n.distance:
                good_matches.append(m)
        # img_cur_kpt = cv2.drawKeypoints(img_cur, kpt_cur, None, (255, 0, 0), 4)
        # cv2.imshow('img2', img2)
        # cv2.waitKey(0)

        if len(good_matches) > min_match_count_thres:

            src_pts = np.float32([kpt_cur[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kpt_st[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            homo_mat, pts_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            homo_mat_old = homo_mat

        else:
            print(f'Not enough matches are found - {len(good_matches)}/{min_match_count_thres}.')
            homo_mat = homo_mat_old

        img_cur_warped = cv2.warpPerspective(img_cur, homo_mat, (img_cur.shape[1], img_cur.shape[0]))
        img_cur_warped_path = os.path.join(out_img_folder, img_list[i])
        cv2.imwrite(img_cur_warped_path, img_cur_warped)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=pts_mask.ravel().tolist(),  # draw only inliers
                           flags=2)
        matches_img = cv2.drawMatches(img_st, kpt_st, img_cur, kpt_cur, good_matches, None, **draw_params)
        tmp_img_path = 'tmp/matches_img_' + img_list[i]
        cv2.imwrite(tmp_img_path, matches_img)

        # cv2.imshow('matches', img3)
        # cv2.waitKey(0)


def preprocessing_avg(img_folder, mask_folder, align_img_folder, out_img_folder, out_mask_folder, roi_num=4):

    if not os.path.exists(out_img_folder):
        os.makedirs(out_img_folder)

    img_list = os.listdir(align_img_folder)
    img_list.sort(key=lambda x: (len(x), x))


    for i in tqdm(range(1, len(img_list))):

        img_avg = cv2.imread(os.path.join(align_img_folder, img_list[i])).astype(np.float32)

        idx_st = max(0, i - 2)
        idx_ed = min(len(img_list), i + 3)

        for j in range(idx_st, idx_ed):
            img_avg += cv2.imread(os.path.join(align_img_folder, img_list[j])).astype(np.float32)

        img_avg /= (idx_ed - idx_st + 1)

        cv2.imwrite(os.path.join(out_img_folder, img_list[i]), img_avg.astype(np.uint8))
        # cv2.imshow('avg', img_avg)
        # cv2.waitKey(0)




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Estimate water level.')
    parser.add_argument(
        '--video-name', type=str, required=True,
        help='Video name.')
   
    args = parser.parse_args()

    print('Args:', args)
    # video_name = 'boston_harbor_20190119'

    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')
    dataset_folder = cfg['paths']['dataset_ubuntu']

    img_folder = os.path.join(dataset_folder, 'test_videos', args.video_name)
    mask_folder = os.path.join(dataset_folder, 'test_annots', args.video_name)

    align_img_folder = os.path.join(dataset_folder, 'test_videos', args.video_name + '_align')

    out_img_folder = os.path.join(dataset_folder, 'test_videos', args.video_name + '_avg')
    out_mask_folder = os.path.join(dataset_folder, 'test_annots', args.video_name + '_avg')

    preprocessing_avg(img_folder, mask_folder, align_img_folder, out_img_folder, out_mask_folder)
