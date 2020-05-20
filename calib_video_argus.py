import os
import argparse
import configparser
import numpy as np
import cv2

window_name = 'first frame'
pts_selected = []
pts_n = 10 + 1
loop_flag = True


def pts_dist(p0, p1):
    return ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5


def mouse_click(event, x, y, flags, param):
    global pts_selected, loop_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        pts_selected.append((x, y))

    if event == cv2.EVENT_LBUTTONUP:

        cv2.circle(param, pts_selected[-1], 5, (0, 0, 200), -1)
        cv2.imshow(window_name, param)

        if len(pts_selected) == pts_n:
            loop_flag = False


def select_ref_pts(img):
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click, param=img.copy())

    cv2.imshow(window_name, img)
    while loop_flag:
        cv2.waitKey(1)

    cv2.destroyWindow(window_name)

    np.save(ref_pts_path, pts_selected)
    print(f'Save ref pts to {ref_pts_path}')
    return pts_selected


def draw_lines(img, pts, lines):
    h, w = img.shape[:2]
    for i in range(pts.shape[0]):
        cv2.circle(img, tuple(pts[i]), 10, (200, 0, 0), -1)

    for i in range(lines.shape[0]):
        cv2.line(
            img,
            (0, int(-lines[i, 2] / lines[i, 1])),
            (w, int(-(lines[i, 2] + lines[i, 0] * w) / lines[i, 1])),
            (255, 255, 255)
        )


def est_img_mat(img, pts_src):

    h, w, c = img.shape
    imsize = (w, h)
    global pts_n

    pts_n_per_line = int(pts_n // 2)
    target_x_d = pts_dist(pts_src[0], pts_src[2])
    target_y_d = pts_dist(pts_src[0], pts_src[1])
    pts_target = [list(pts_src[0])]
    pts_target.append([pts_src[0][0], pts_src[0][1] + target_y_d])

    for i in range(1, pts_n_per_line):
        new_p = [pts_src[0][0] - i * target_x_d, pts_src[0][1]]
        pts_target.append(new_p)
        new_p = [pts_src[0][0] - i * target_x_d, pts_src[0][1] + target_y_d]
        pts_target.append(new_p)

    # Overlap point
    pts_target.append(pts_target[pts_n_per_line - 1])

    pts_n = 10
    pts_src = pts_src[:pts_n]
    pts_target = pts_target[:pts_n]

    print('Point dst:', pts_target)

    pts_src = np.float32(pts_src)
    pts_target = np.float32(pts_target)
    f_mat, mat_mask = cv2.findFundamentalMat(pts_src, pts_target, cv2.FM_LMEDS)

    print(f_mat, mat_mask)
    #
    # ret_val, homo_mat1, homo_mat2 = cv2.stereoRectifyUncalibrated(pts_src, pts_target, f_mat, imsize)
    # print(ret_val, homo_mat1, homo_mat1)
    #
    # K = np.eye(3, dtype=np.float64)
    # LeftR = np.linalg.inv(K) * homo_mat1 * K
    # map1, map2 = cv2.initUndistortRectifyMap(K, None, LeftR, K, (w, h), cv2.CV_16SC2)
    # img_remaped = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

    img2 = cv2.warpPerspective(img, f_mat, imsize)
    # img2 = np.zeros_like(img)

    l_src = cv2.computeCorrespondEpilines(pts_target, 2, f_mat)
    l_target = cv2.computeCorrespondEpilines(pts_src, 1, f_mat)

    draw_lines(img, pts_src, l_src.reshape(-1, 3))
    draw_lines(img2, pts_target, l_target.reshape(-1, 3))

    img = cv2.resize(img, None, None, 0.4, 0.4)
    img2 = cv2.resize(img2, None, None, 0.4, 0.4)
    # img_remaped = cv2.resize(img_remaped, None, None, 0.4, 0.4)
    # img_warped = cv2.resize(img_warped, None, None, 0.5, 0.5)
    cv2.imshow('img', img)
    # cv2.imshow('img_remapped', img_remaped)
    cv2.imshow('img2', img2)
    # cv2.imshow('warped', img_warped)
    cv2.waitKey()

    return f_mat


def calib_video(video_folder, st_idx, ref_pts_path, recalib_flag):
    img_list = os.listdir(video_folder)
    img_list.sort(key=lambda x: (len(x), x))
    img_st = cv2.imread(os.path.join(video_folder, img_list[st_idx]))

    if recalib_flag:
        pts_src = select_ref_pts(img_st)
    else:
        try:
            print(f'Load ref pts from {ref_pts_path}')
            pts_src = np.load(ref_pts_path)
            if len(pts_src) != pts_n:
                print(f'{len(pts_src)} is not equal to {pts_n}')
                pts_src = select_ref_pts(img_st)
        except FileNotFoundError as e:
            print(f'ref_pts_path: {ref_pts_path} doesn\'t exist. Reselect the reference points.')
            pts_src = select_ref_pts(img_st)

    print('Point src:', pts_src)

    homo_mat = est_img_mat(img_st, pts_src)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calibrate video.')
    parser.add_argument(
        '--video-name', type=str, required=True,
        help='Video name.')
    parser.add_argument(
        '--st', type=int, default=2,
        help='Video name.')
    parser.add_argument(
        '--recalib', action='store_true',
        help='Recalibate the video')
    args = parser.parse_args()

    print('Args:', args)

    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    root_dir = '/Ship01/Dataset/water/Argus_videos/C4_Timex'

    video_folder = os.path.join(root_dir, args.video_name)

    ref_pts_dir = os.path.join(root_dir, 'ref_pts')
    if not os.path.exists(ref_pts_dir):
        os.makedirs(ref_pts_dir)
    ref_pts_path = os.path.join(ref_pts_dir, args.video_name + '.npy')

    calib_video(video_folder, args.st, ref_pts_path, args.recalib)
