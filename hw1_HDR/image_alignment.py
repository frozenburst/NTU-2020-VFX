# paulolbear@cmlab.csie.ntu.edu.tw at 2020/04/05
# Image alignment for HDR homework of VFX NTU 2020
from pathlib import Path

import cv2
import os.path as op
import matplotlib.pyplot as plt
import numpy as np

data_pth = op.join("images", "test")
img_set = []
offset_sets = []

def imshow(img, cmap=None):
    plt.imshow(img, cmap)
    plt.show()

def collect_image():
    print("Collecting Image with in path:", data_pth)
    img_set.clear()
    for img_pth in sorted(Path(data_pth).glob(f'*.JPG')):
        print(img_pth)
        img = cv2.imread(op.join(img_pth))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #imshow(img)
        img_set.append(img)

def img_shift(img, x, y):
    img_size = img.shape
    M = np.float32([[1, 0, x], [0, 1, y]])
    return cv2.warpAffine(img, M, (img_size[1], img_size[0]))

def offset_sets_initialize():
    offset_sets.clear()
    range_offset = [-1, 0, 1]
    for x in range_offset:
        for y in range_offset:
            offset_sets.append([x, y])
    print(offset_sets)

def pyramid_compare(img, level):
    # Not reach top of pyramid, recursive call
    last_offset = [0, 0]
    if level < pyramid_level-1:
        img_resize = cv2.resize(img, (round(img.shape[1]/2), round(img.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
        last_offset = pyramid_compare(img_resize, level+1)
    
    threshold = np.median(img)
    _, img_thresh = cv2.threshold(img, threshold, 1, cv2.THRESH_BINARY)

    lowest_cost = img.shape[0] * img.shape[1]
    for offset in offset_sets:
        img_base, mask = img_base_set[level]
        test_offset = [last_offset[0] * 2 + offset[0], last_offset[1] * 2 + offset[1]]
        img_thresh_shifted = img_shift(img_thresh, test_offset[0], test_offset[1])
        # XOR with base, and not on the mask part
        cost_map = np.logical_and(np.logical_xor(img_base, img_thresh_shifted), np.logical_not(mask))
        cost = np.sum(cost_map)
        #print(last_offset, offset)
        #print(test_offset, cost, level)
        if cost < lowest_cost:
            lowest_cost = cost
            best_offset = test_offset
    #print(best_offset, lowest_cost)
    return best_offset

if __name__ == "__main__":
    collect_image()
    # Alignment: Median Threshold Bitmap
    # Take first image as base to alignment
    img_name = op.join(data_pth, f'00.JPG')
    cv2.imwrite(img_name, img_set[0])
    print("Base images:", img_name)

    img_set_copy = img_set.copy()
    img_base = img_set_copy[0]
    img_set_copy.remove(img_base)
    pyramid_level = 6

    img_base_set = []
    for i in range(pyramid_level):
        img_gray = cv2.cvtColor(img_base, cv2.COLOR_RGB2GRAY)
        threshold = np.median(img_gray)
        _, img_thresh = cv2.threshold(img_gray, threshold, 1, cv2.THRESH_BINARY)
        mask_img = cv2.inRange(img_gray, threshold - 10, threshold + 10) / 255
        img_base_set.append((img_thresh, mask_img))
        #imshow(img_thresh, 'gray')
        #imshow(mask_img, 'gray')

        img_base = cv2.resize(img_base, (round(img_base.shape[1]/2), round(img_base.shape[0]/2)), interpolation=cv2.INTER_CUBIC)
    
    offset_sets_initialize()

    print("Start calculate the offset of each image")
    for img_index in range(len(img_set_copy)):
        tmp_img = img_set_copy[img_index]
        tmp_img_gray = cv2.cvtColor(tmp_img, cv2.COLOR_RGB2GRAY)
        offset = pyramid_compare(tmp_img_gray, 0)
        img_shifted = img_shift(tmp_img, offset[0], offset[1])
        img_name = op.join(data_pth, f'0{img_index+1}.JPG')
        cv2.imwrite(img_name, img_shifted)
        print(offset, img_name)