import os
import argparse
import configparser
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
        print('Point list:', pts)
        cv2.circle(param, pts[-1], 5, (0, 0, 200), -1)
        cv2.imshow(window_name, param)
        
        if len(pts) == pts_n:
            loop_flag = False

def calib_video(video_folder):
    
    img_list = os.listdir(video_folder)
    img_list.sort(key = lambda x: (len(x), x))
    img_st = cv2.imread(os.path.join(video_folder, img_list[0]))

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click, param = img_st)

    cv2.imshow(window_name, img_st)
    while loop_flag:
        cv2.waitKey(1)

    
    



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