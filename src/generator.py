import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from glob import glob

import cv2

from src import utils
from src.vector import Vector


class Generator:

    def __init__(self, img_path):

        self.img = cv2.imread(img_path)
        self.img_size = self.img.shape[:2] # Height, Width, Channel
        self.aspect_ratio = self.img_size[0] / self.img_size[1]
        
        self.training_G_Net_folder = 'data/training_G_Net/'
        if not os.path.exists(self.training_G_Net_folder):
            os.mkdir(self.training_G_Net_folder)

    def get_smooth_curve(self, x_len, x_pt_n, x_offset, y_offset, x_step, smooth_flag=False):
        
        x_arr = []
        y_arr = []
        
        for i in range(x_pt_n+1):
            
            if i == 0:
                x = 0
            elif i == x_pt_n:
                x = x_len - 1
            else:
                x = round(x_step * i + random.uniform(-x_offset, x_offset))
            y = round(random.uniform(-y_offset, y_offset))

            x_arr.append(x)
            y_arr.append(y)

        x_arr = list(set(x_arr))
        y_arr = y_arr[:len(x_arr)]
        x_arr.sort()

        if smooth_flag:
            smooth_func = interpolate.interp1d(x_arr, y_arr, kind='cubic')
            x_arr = np.arange(0, x_len, dtype=np.int32)
            y_arr = smooth_func(x_arr).astype(np.int32)

        # plt.plot(x_arr, y_arr, 'r')
        # plt.plot(x_arr_s, y_arr_s, 'b')
        # plt.show()

        return x_arr, y_arr


    def get_mask(self, offset_rate_h, offset_rate_w):

        piece_h = self.img_size[0] / self.h_n
        piece_w = self.img_size[1] / self.w_n

        offset_h = piece_h * offset_rate_h
        offset_w = piece_w * offset_rate_w

        self.mask = utils.new_array(self.img_size, 0)

        # Vertical cuts
        for i in range(1, self.w_n):

            x_arr, y_arr = self.get_smooth_curve(self.img_size[0], self.h_n, offset_h, offset_w, piece_h, True)
            y_arr = y_arr + round(i * piece_w)
            y_arr = np.clip(y_arr, 0, self.img_size[1] - 1)

            for j in range(self.img_size[0]):
                self.mask[x_arr[j]][y_arr[j]] = 255
                if j > 0:
                    st = min(y_arr[j - 1], y_arr[j])
                    ed = max(y_arr[j - 1], y_arr[j])
                    for k in range(st, ed + 1):
                        self.mask[x_arr[j]][k] = 255

        # Horizontal cuts
        for i in range(1, self.h_n):

            x_arr, y_arr = self.get_smooth_curve(self.img_size[1], self.w_n, offset_w, offset_h, piece_w, True)
            y_arr = y_arr + round(i * piece_h)
            y_arr = np.clip(y_arr, 0, self.img_size[0] - 1)

            for j in range(self.img_size[1]):
                self.mask[y_arr[j]][x_arr[j]] = 255
                if j > 0:
                    st = min(y_arr[j - 1], y_arr[j])
                    ed = max(y_arr[j - 1], y_arr[j])
                    for k in range(st, ed + 1):
                        self.mask[k][x_arr[j]] = 255

        cv2.imwrite('tmp/mask_init.png', np.array(self.mask, dtype=np.uint8))
        # cv2.imshow('mask', self.mask)
        # cv2.waitKey()

    def get_regions(self, small_region_area_ratio):
        
        dirs = [Vector(0,-1), Vector(0, 1), Vector(-1, 0), Vector(1, 0)] # (x, y)
        small_region_area_limit = small_region_area_ratio * \
            self.img_size[0] * self.img_size[1] / (self.w_n * self.h_n)

        # self.region_mat = utils.new_array(self.img_size, -1)
        # self.region_cnt = 0

        # small_regions = []
        # small_region_flags = []
        
        # for y in range(self.img_size[0]):
        #     for x in range(self.img_size[1]):
                
        #         if self.mask[y][x] != 0 or self.region_mat[y][x] != -1:
        #             continue
                
        #         self.region_mat[y][x] = self.region_cnt
        #         que = []
        #         que.append(Vector(x, y))
        #         front_idx = 0

        #         while front_idx < len(que):

        #             cur_p = que[front_idx]
        #             front_idx += 1

        #             for dir in dirs:
        #                 next_p = cur_p + dir
        #                 if utils.check_outside(next_p.x, next_p.y, self.img_size[1], self.img_size[0]) or \
        #                     self.mask[next_p.y][next_p.x] != 0 or self.region_mat[next_p.y][next_p.x] != -1:
        #                     continue
        #                 self.region_mat[next_p.y][next_p.x] = self.region_cnt
        #                 que.append(next_p)

        #         if len(que) < small_region_area_limit:
        #             small_regions.append(que)
        #             small_region_flags.append(True)
        #         else:
        #             small_region_flags.append(False)

        #         self.region_cnt += 1
        mask = np.invert(np.array(self.mask, dtype=np.uint8))

        self.region_cnt, self.region_mat, stats, centroids = \
            cv2.connectedComponentsWithStats(mask, connectivity=4, ltype=cv2.CV_32S)
        stats = stats.tolist()

        # Remap region idx
        region_idx_map = -1 * np.ones(self.region_cnt, dtype=np.int32)
        region_new_cnt = 0

        for i in range(1, self.region_cnt):
            if stats[i][4] < small_region_area_limit:
                region_idx_map[i] = -1
            else:
                region_idx_map[i] = region_new_cnt
                region_new_cnt += 1
        
        self.region_mat = region_idx_map[self.region_mat]
        
        print('Region cnt old:', self.region_cnt - 1, 'Region cnt new:', region_new_cnt)
        self.region_cnt = region_new_cnt
        
        # Expand valid region to fill out the canvas
        bg_pts = np.transpose(np.nonzero(self.region_mat == -1)).tolist()
        self.region_mat = self.region_mat.tolist()
        que = []

        for bg_pt in bg_pts:
            cur_p = Vector(bg_pt[1], bg_pt[0])
            for dir in dirs:
                next_p = cur_p + dir
                if utils.check_outside(next_p.x, next_p.y, self.img_size[1], self.img_size[0]) or \
                    self.region_mat[next_p.y][next_p.x] == -1:
                    continue
                que.append(next_p)
        
        while len(que) > 0:
            cur_p = que.pop(0)
            for dir in dirs:
                next_p = cur_p + dir
                if utils.check_outside(next_p.x, next_p.y, self.img_size[1], self.img_size[0]) or \
                    self.region_mat[next_p.y][next_p.x] != -1:
                    continue
                self.region_mat[next_p.y][next_p.x] = self.region_mat[cur_p.y][cur_p.x]
                que.append(next_p)

        # Check the region mat
        unlabel_pts = np.transpose(np.nonzero(np.ma.masked_equal(self.region_mat, -1).mask))
        print('Number of unlabel points: ', unlabel_pts.size)

        
        # for i in range(self.region_cnt):
        #     mask = np.ma.masked_equal(self.region_mat, i).mask.astype(np.uint8)
        #     mask = mask * 255
        #     cv2.imwrite('tmp/' + str(i) + '.png', mask)
        #     cv2.imshow('tmp', mask)
        #     cv2.waitKey(0)
    

    def save_regions(self, iter, save_whole_regions_flag):

        if save_whole_regions_flag:

            file_path = os.path.join(self.training_G_Net_folder, '%d.npy' % iter)
            np.save(file_path, np.array(self.region_mat, dtype=np.int32))

            f = open(file_path[:-3] + 'txt', 'w')
            f.write(str(self.region_cnt))
            f.close()
            print('Save to %s / %d.txt' % (file_path, iter))


    def run(self, piece_n, sample_n, offset_rate_h=0.2, offset_rate_w=0.2, small_region_area_ratio=0.25, save_whole_regions_flag=True):
        
        self.piece_n = piece_n
        self.w_n = math.floor(math.sqrt(piece_n / self.aspect_ratio))
        self.h_n = math.floor(self.w_n * self.aspect_ratio)

        print('Hori pieces:', self.w_n, 'Vert pieces:', self.h_n)
        print('Offset rate h: %.2f, w: %.2f' % (offset_rate_h, offset_rate_w))
        print('Small region area ratio: %.2f' % small_region_area_ratio)

        data_st_idx = len(glob(os.path.join(self.training_G_Net_folder, '*.npy')))

        for i in range(sample_n):
            self.get_mask(offset_rate_h, offset_rate_w)
            self.get_regions(small_region_area_ratio)
            self.save_regions(data_st_idx + i, save_whole_regions_flag)

    