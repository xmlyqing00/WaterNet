import os
import argparse
import numpy as np
import cv2

def cvt_object_label(img, ori_label_color, dst_label_color=None):

    if dst_label_color is None:
        dst_label_color = ori_label_color

    height, width, channels = img.shape
    blue_mask = np.ones([height, width, 1], dtype=np.uint8) * ori_label_color[0]
    green_mask = np.ones([height, width, 1], dtype=np.uint8) * ori_label_color[1]
    red_mask = np.ones([height, width, 1], dtype=np.uint8) * ori_label_color[2]
    color_mask = cv2.merge((blue_mask, green_mask, red_mask))

    # Set label_color to 0
    img = cv2.bitwise_xor(img, color_mask)

    # Set label_color to 255
    img = cv2.bitwise_not(img)

    # Set other pixel to 0
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(mask, np.array(dst_label_color))

    return img

if __name__ == '__main__':

    video_name = 'holmrook'
    folder = os.path.join('/Ship01/Dataset/water/collection/test_annots/', video_name)
    
    name_list = os.listdir(folder)
    for frame_name in name_list:
        if frame_name[-5:] == '_json' and frame_name[0] != '.':
            print(frame_name)

            label = cv2.imread(os.path.join(folder, frame_name, 'label.png'))
            label_w = cvt_object_label(label, (0, 0, 128), (255, 255, 255))

            cv2.imwrite(os.path.join(folder, '%s.png' % frame_name[:-5]), label_w)


    parser = argparse.ArgumentParser(description='Convert Object Label to W/B')
    parser.add_argument(
        '-v', type=str, default=None, help='Test video name (default: none).')
    parser.add_argument(
        '-v', type=str, default=None, help='Test video name (default: none).')
    parser.add_argument(
        '--no-temporal', action='store_true',
        help='Evaluate the video without temporally updating templates.')
    parser.add_argument(
        '--no-conf', action='store_true',
        help='Evaluate the video without high-confidence features updating templates.')
    parser.add_argument(
        '--no-aa', action='store_true',
        help='Evaluate the video without appearance-adaptive branch.')
    parser.add_argument(
        '-b', '--benchmark', action='store_true',
        help='Evaluate the video with groundtruth.')
    parser.add_argument(
        '-c', '--checkpoint', default=None, type=str, metavar='PATH',
        help='Path to latest checkpoint (default: none).')
    parser.add_argument(
        '-v', '--video-name', default=None, type=str,
        help='Test video name (default: none).')
    parser.add_argument(
        '-m', '--model-name', default='AANet', type=str,
        help='Model name for the ouput segmentation, it will create a subfolder under the out_folder.')
    parser.add_argument(
        '-o', '--out-folder', default=cfg['paths'][cfg_dataset], type=str, metavar='PATH',
        help='Folder for the output segmentations.')
    args = parser.parse_args()

    print('Args:', args)

    if args.checkpoint is None:
        raise ValueError('Must input checkpoint path.')
    if args.video_name is None:
        raise ValueError('Must input video name.')