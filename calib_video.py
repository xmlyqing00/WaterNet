import os
import argparse
import configparser
import numpy as np
import cv2

window_name = 'first frame'
pts = []
pts_n = 4
loop_flag = True

def mouse_click(event, x, y, flags, param):
    
    global pts, loop_flag

    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x,y))

    if event == cv2.EVENT_LBUTTONUP:
        
        cv2.circle(param, pts[-1], 5, (0, 0, 200), -1)
        cv2.imshow(window_name, param)
        
        if len(pts) == pts_n:
            loop_flag = False

def calib_img(img):

    global pts

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click, param = img.copy())

    cv2.imshow(window_name, img)
    while loop_flag:
        cv2.waitKey(1)

    print('Point src:', pts)

    d_x = ((pts[1][0] - pts[0][0]) ** 2 + (pts[1][1] - pts[0][1]) ** 2) ** 0.5
    d_y = ((pts[2][0] - pts[0][0]) ** 2 + (pts[2][1] - pts[0][1]) ** 2) ** 0.5
    pts_t = [pts[0]]
    pts_t.append((pts_t[0][0] + d_x, pts_t[0][1]))
    pts_t.append((pts_t[0][0], pts_t[0][1] + d_y))
    pts_t.append((pts_t[0][0] + d_x, pts_t[0][1] + d_y))

    print('Point dst:', pts_t)

    pts = np.float32(pts)
    pts_t = np.float32(pts_t)
    homo_mat, _ = cv2.findHomography(pts, pts_t)   
    # print(homo_mat) 
    
    img_warped = cv2.warpPerspective(img, homo_mat, (img.shape[1], img.shape[0]))
    cv2.imshow('warped', img_warped)
    cv2.waitKey()


def calib_video(video_folder):
    
    img_list = os.listdir(video_folder)
    img_list.sort(key = lambda x: (len(x), x))
    img_st = cv2.imread(os.path.join(video_folder, img_list[0]))

    calib_img(img_st)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Calibrate video.')
    parser.add_argument(
        '--video-name', type=str, required=True,
        help='Video name.')
    parser.add_argument(
        '--recalib', action='store_true',
        help='Recalibate the video')
    args = parser.parse_args()

    print('Args:', args)

    # Paths
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    video_folder = os.path.join(cfg['paths']['dataset_ubuntu'], 'test_videos', args.video_name)
    calib_video(video_folder)