import os
import math
import numpy as np
from glob import glob
import cv2

from src.generator import Generator
from src import utils


def get_boundary_map(region_mat):

    region_mat = np.int16(region_mat)
    grad_x = cv2.Sobel(region_mat, cv2.CV_16S, 1, 0, ksize=1)
    grad_y = cv2.Sobel(region_mat, cv2.CV_16S, 0, 1, ksize=1)
    div = abs(grad_x) + abs(grad_y)
    ret, boundary = cv2.threshold(div, 0, 255, cv2.THRESH_BINARY)
    boundary = np.uint8(boundary)
    boundary_map = cv2.cvtColor(boundary, cv2.COLOR_GRAY2BGR)

    # cv2.imwrite('tmp/boundary_%d.png' % geo_regions_idx, boundary)
    # cv2.imshow('boundary', boundary)
    # cv2.waitKey()

    return boundary_map


def generate_geo_patches(patch_size=(48, 48)):
    
    geo_regions_folder = 'data/geo_regions/'
    geo_regions_size = len(glob(os.path.join(geo_regions_folder, '*.npy')))
    geo_patches_folder = 'data/geo_patches/'
    geo_patches_idx = len(glob(os.path.join(geo_patches_folder, '*.png')))

    for geo_regions_idx in range(geo_regions_size):

        print('Loading traininig data: [%4d / %4d]' % (geo_regions_idx, geo_regions_size))

        f = open(os.path.join(geo_regions_folder, '%d.txt' % geo_regions_idx), 'r')
        region_cnt = int(f.read())
        f.close()

        region_mat = np.int16(np.load(os.path.join(geo_regions_folder, '%d.npy' % geo_regions_idx)))
        boundary_map = get_boundary_map(region_mat)
        cv2.imwrite(os.path.join(geo_regions_folder, '%d.png' % geo_regions_idx), boundary_map)
        target_angle = math.degrees(math.pi / 2)

        for region_idx in range(region_cnt):

            region = np.uint8(np.ma.masked_equal(region_mat, region_idx).mask) * 255
            region = cv2.GaussianBlur(region, (9, 9), 0)            

            corners = cv2.goodFeaturesToTrack( \
                region, \
                maxCorners=0, \
                qualityLevel=0.05, \
                minDistance=10, \
                blockSize=5, \
                useHarrisDetector=False)
            grad_x = cv2.Sobel(region, cv2.CV_16S, 1, 0, ksize=5)
            grad_y = cv2.Sobel(region, cv2.CV_16S, 0, 1, ksize=5)

            for pt in corners:
                x, y = pt.ravel()
                x = int(x)
                y = int(y)
                angle = math.degrees(math.atan2(grad_y[y, x], grad_x[y, x]))
                rot_angle = angle - target_angle

                M = cv2.getRotationMatrix2D((x, y), rot_angle, 1)
                rotated_mat = cv2.warpAffine(region, M, (region.shape[1], region.shape[0]), cv2.INTER_CUBIC)
                
                patch = cv2.getRectSubPix(rotated_mat, patch_size, (x, y))
                cv2.imwrite(os.path.join(geo_patches_folder, '%d.png' % geo_patches_idx), patch)
                geo_patches_idx += 1

                # cv2.circle(rotated_mat, (x, y), 3, (0, 0, 200), -1, lineType=8)
                # cv2.imshow('patch', patch)
                # cv2.imshow('region', region)
                # cv2.imshow('rotated', rotated_mat)
                # cv2.waitKey(0)
        
       
def generate_geo_regions():
    img_path = 'data/gt/3.png'
    piece_n = 100
    generator = Generator(img_path)
    generator.run(piece_n, sample_n = 10, offset_rate_h=1, offset_rate_w=1, small_region_area_ratio=0.25)


if __name__ == '__main__':
    generate_geo_patches()