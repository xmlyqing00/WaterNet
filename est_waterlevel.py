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
from pandas.plotting import register_matplotlib_converters
        

from calib_video import get_video_homo

seg_thres = 128

time_fmt = mdates.DateFormatter('%M/%d %H:%M')
# print(time_arr)

tick_spacing = 4
# ticker_locator = ticker.MultipleLocator(tick_spacing)
ticker_locator = mdates.HourLocator(interval=tick_spacing)

register_matplotlib_converters()

def track_object(img_folder, overlay_folder, out_folder, homo_mat, sample_iter=0, bbox_st=None):

    tracker = cv2.TrackerCSRT_create()

    img_list = os.listdir(img_folder)
    img_list.sort(key = lambda x: (len(x), x))

    overlay_list = os.listdir(overlay_folder)
    overlay_list.sort(key = lambda x: (len(x), x))

    # First frame
    frame_st = cv2.imread(os.path.join(img_folder, img_list[0]))
    overlay_st = cv2.imread(os.path.join(overlay_folder, overlay_list[0]))
    
    fx = overlay_st.shape[1] / frame_st.shape[1]
    fy = overlay_st.shape[0] / frame_st.shape[0]
    scale_mat = np.matrix([[fx, 0, 0], [0, fy, 0], [0, 0, 1]])

    homo_img_mat = scale_mat * np.asmatrix(homo_mat)
    homo_out_mat = homo_img_mat * scale_mat.I
    
    # frame_st = cv2.resize(frame_st, (overlay_st.shape[1], overlay_st.shape[0]))

    img_size = (overlay_st.shape[1], overlay_st.shape[0])
    frame_st = cv2.warpPerspective(frame_st, homo_img_mat, img_size)
    if sample_iter == 0:
        overlay_st = cv2.warpPerspective(overlay_st, homo_out_mat, img_size)
    
    # bbox_st = (768, 223, 46, 42)
    # Boston harbor bbox =  ((241, 38, 22, 10), (309, 52, 34, 22), (516, 77, 13, 35))
    if bbox_st is None:
        while True:    
            bbox_st = cv2.selectROI('First Frame', frame_st, fromCenter=False)
            if bbox_st[2] > 0 and bbox_st[3] > 0:
                break
    x, y, w, h = [int(v) for v in bbox_st]
    cv2.rectangle(overlay_st, (x, y), (x+w, y + h), (0, 200, 0), 2)
    out_path = os.path.join(out_folder, img_list[0][:-4] + '_tag.png')
    cv2.imwrite(out_path, overlay_st)

    print('Init', bbox_st)

    tracker.init(frame_st, bbox_st)
    bbox_old = bbox_st


    key_pts = [(int(x + w/2), int(y + h))]

    for i in range(1, len(img_list)):

        img = cv2.imread(os.path.join(img_folder, img_list[i]))
        overlay = cv2.imread(os.path.join(overlay_folder, overlay_list[i]))
        
        # img = cv2.resize(img, (overlay_st.shape[1], overlay_st.shape[0]))
        img = cv2.warpPerspective(img, homo_img_mat, img_size)
        if sample_iter == 0:
            overlay = cv2.warpPerspective(overlay, homo_out_mat, img_size)

        op_flag, bbox = tracker.update(img)

        # print(i, op_flag, bbox)
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
    
    return key_pts, homo_out_mat, bbox_st
        

def est_water_boundary(seg_folder, out_folder, key_pts, homo_mat):

    seg_list = os.listdir(seg_folder)
    seg_list.sort(key = lambda x: (len(x), x))

    overlay_list = os.listdir(out_folder)
    overlay_list.sort(key = lambda x: (len(x), x))

    water_level_px = np.zeros(len(key_pts))

    for i in range(len(key_pts)):
        
        seg_path = os.path.join(seg_folder, seg_list[i])
        seg = cv2.imread(seg_path)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
        seg = cv2.warpPerspective(seg, homo_mat, (seg.shape[1], seg.shape[0]))

        if i > 0:
            water_level_px[i] = water_level_px[i - 1]

        for y in range(key_pts[i][1] + 1, seg.shape[0]):
            if seg[y][key_pts[i][0]] > seg_thres:
                water_level_px[i] = y - key_pts[i][1]

                out_path = os.path.join(out_folder, overlay_list[i])          
                overlay = cv2.imread(out_path)
                cv2.line(overlay, key_pts[i], (key_pts[i][0], y), (0, 0, 200), 2)
                cv2.imwrite(out_path, overlay)
                # cv2.imshow('overlay', overlay)
                # cv2.waitKey()

                break
    
    return water_level_px


def get_time_arr(img_folder):

    img_list = os.listdir(img_folder)
    img_list.sort(key = lambda x: (len(x), x))

    time_arr = []

    for img_name in img_list:
        year = img_name[-17:-13]
        mon = img_name[-20:-18]
        day = img_name[-23:-21]
        timestamp = f'{year}-{mon}-{day} ' + img_name[-12:-4].replace('-', ':')
        timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        time_arr.append(timestamp)
    
    return time_arr


def est_waterlevel(video_name, dataset_folder, recalib_flag, reref_flag, sample_times=1, model_name='WaterNet'):

    img_folder = os.path.join(dataset_folder, 'test_videos', video_name)
    overlay_folder = os.path.join(dataset_folder, 'results', model_name + '_overlays', video_name)
    out_folder = os.path.join(dataset_folder, 'results', model_name + '_overlays_bbox', video_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Calibrate the video
    homo_mat_folder = os.path.join(dataset_folder, 'test_videos/homo_mats')
    if not os.path.exists(homo_mat_folder):
        os.makedirs(homo_mat_folder)
    homo_mat_path = os.path.join(homo_mat_folder, args.video_name + '.npy')
    homo_mat = get_video_homo(img_folder, homo_mat_path, recalib_flag)

    time_arr = get_time_arr(img_folder)

    water_level_folder = os.path.join(dataset_folder, 'water_level', model_name, video_name)
    if not os.path.exists(water_level_folder):
        os.makedirs(water_level_folder)
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    ref_bbox_path = os.path.join(water_level_folder, 'ref_bbox.npy')
    if not reref_flag:
        print('Load bbox of the reference objects.', ref_bbox_path)
        ref_bbox = np.load(ref_bbox_path)
        sample_times = ref_bbox.shape[0]
    else:
        ref_bbox = []

    for sample_iter in range(sample_times):
        
        print(f'Estimate water level by reference {sample_iter}')

        if sample_iter > 0:
            overlay_folder = out_folder
        # Get points of reference objs
        if reref_flag:
            key_pts, homo_out_mat, bbox_st = track_object(img_folder, overlay_folder, out_folder, homo_mat, sample_iter)
            ref_bbox.append(bbox_st)
        else:
            key_pts, homo_out_mat, bbox_st = track_object(img_folder, overlay_folder, out_folder, homo_mat, sample_iter, tuple(ref_bbox[sample_iter]))
        
        seg_folder = os.path.join(dataset_folder, 'results', model_name + '_segs', video_name)
        water_level_px = est_water_boundary(seg_folder, out_folder, key_pts, homo_out_mat)
        water_level_px = -(water_level_px - water_level_px[0])
        # print(water_level_px)

        if sample_iter == 0:
            water_level_px_all = water_level_px
        else:
            water_level_px_all = np.vstack((water_level_px_all, water_level_px))

        ax.plot(time_arr, water_level_px, '-', label=f'By ref {sample_iter+1} (px)')
        # plt.gca().xaxis.set_major_formatter(time_fmt)
        # plt.show()

        # water_level_path = os.path.join(water_level_folder, f'ref{sample_iter}.png')
        # plt.savefig(water_level_path, dpi=300)

    if reref_flag:
        np.save(ref_bbox_path, np.array(ref_bbox))

    if sample_times > 1:
        ax.plot(time_arr, water_level_px_all.mean(0), label=f'Avg (px)')
    else:
        water_level_px_all = np.expand_dims(water_level_px_all, axis=0)
    water_level_path = os.path.join(water_level_folder, 'water_level_px.npy')
    np.save(water_level_path, water_level_px_all)

    time_arr_path = os.path.join(water_level_folder, 'time_arr.npy')
    np.save(time_arr_path, np.array(time_arr))

    ax.xaxis.set_major_formatter(time_fmt)
    ax.xaxis.set_major_locator(ticker_locator)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc='lower right')

    water_level_path = os.path.join(water_level_folder, 'water_level_px.png')
    fig.savefig(water_level_path, dpi=300)

    print('Save water_level_px.npy and water_level_px.png')


def plot_hydrograph(water_level_folder):

    print('Load water_level_px.npy and time_arr.npy')

    water_level_path = os.path.join(water_level_folder, 'water_level_px.npy')
    water_level_px_all = np.load(water_level_path)

    time_arr_path = os.path.join(water_level_folder, 'time_arr.npy')
    time_arr_eval = np.load(time_arr_path, allow_pickle=True)

    gt_path = os.path.join(water_level_folder, 'gt.csv')
    if not os.path.exists(gt_path):
        print(f'The groundtruth file doesn\'t exist. {gt_path}')
        return
    gt_csv = pd.read_csv(gt_path)

    gt_csv.iloc[:, 0] = gt_csv.iloc[:, 0] + ' ' + gt_csv.iloc[:, 1]
    time_arr_gt = pd.to_datetime(gt_csv.iloc[:, 0])

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    for i in range(water_level_px_all.shape[0]):
        ax.plot(time_arr_eval, water_level_px_all[i, :], label=f'By ref {i} (px)')

    if water_level_px_all.shape[0] > 1:
        ax.plot(time_arr_eval, water_level_px_all.mean(0), label=f'Avg (px)')

    ax.plot(time_arr_gt, gt_csv.iloc[:, 4], label=f'Groundtruth (ft)')

    ax.xaxis.set_major_locator(ticker_locator)
    ax.xaxis.set_major_formatter(time_fmt)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc='lower right')

    water_level_path = os.path.join(water_level_folder, 'water_level_px_all.png')
    fig.savefig(water_level_path, dpi=300)
    # plt.show()

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111)

    if water_level_px_all.shape[0] > 1:
        ax.plot(time_arr_eval, water_level_px_all.mean(0), label=f'Avg (px)')
    else:
        ax.plot(time_arr_eval, water_level_px_all[0, :], label=f'By ref 0 (px)')

    ax.plot(time_arr_gt, gt_csv.iloc[:, 4], label=f'Groundtruth (ft)')

    ax.xaxis.set_major_formatter(time_fmt)
    ax.xaxis.set_major_locator(ticker_locator)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(loc='lower right')

    water_level_path = os.path.join(water_level_folder, 'water_level_px_cmp.png')
    fig.savefig(water_level_path, dpi=300)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Estimate water level.')
    parser.add_argument(
        '--video-name', type=str, required=True,
        help='Video name.')
    parser.add_argument(
        '--model-name', type=str, default='WaterNet',
        help='Video name.')
    parser.add_argument(
        '--recalib', action='store_true',
        help='Recalibate the video')
    parser.add_argument(
        '--reref', action='store_true',
        help='Re-pick the reference objects in the video')
    parser.add_argument(
        '--samples', type=int, default=1,
        help='Recalibate the video')
    parser.add_argument(
        '--plot', action='store_true',
        help='Recalibate the video')
    args = parser.parse_args()

    print('Args:', args)
    # video_name = 'boston_harbor_20190119'

    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')
    dataset_folder = cfg['paths']['dataset_ubuntu']

    water_level_folder = os.path.join(dataset_folder, 'water_level', args.model_name, args.video_name)

    if args.plot:
        plot_hydrograph(water_level_folder)
    else:
        est_waterlevel(args.video_name, dataset_folder, args.recalib, args.reref, sample_times = args.samples, model_name=args.model_name)
        plot_hydrograph(water_level_folder)

    