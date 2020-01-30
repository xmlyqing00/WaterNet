import os
import configparser

def print_sample_name(video_dir, sample_n):

    print('Video dir:', video_dir)

    img_list = os.listdir(video_dir)
    img_list.sort(key = lambda x: (x, len(x)))
    imgs_n = len(img_list)

    print('%d %s' % (0, img_list[0]))
    print('%d %s' % (1, img_list[1]))

    step = int(imgs_n / (sample_n - 1))
    for i in range(step, imgs_n, step):
        print(i, img_list[i])
    
    print('%d %s' % (imgs_n - 1, img_list[-1]))


if __name__ == '__main__':
    
    cfg = configparser.ConfigParser()
    cfg.read('settings.conf')

    video_name = 'pineapple_willy_clip1'
    root_folder = cfg['paths']['dataset_ubuntu']
    video_dir = os.path.join(root_folder, 'test_videos', video_name)
    annot_dir = os.path.join(root_folder, 'test_annots', video_name)

    sample_n = 5

    print_sample_name(video_dir, sample_n)

    if not os.path.exists(annot_dir):
        os.makedirs(annot_dir)