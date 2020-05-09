from pathlib import Path

import cv2 as cv
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import math
import pdb

def imshow(img, cmap=None):
    plt.imshow(img, cmap)
    plt.show()


def collect_images(pth):
    # In cv2, is BGR instead of RGB
    img_set = []
    img_set.clear()
    for img_pth in sorted(Path(pth).glob(f'prtn*.jpg')):
        print(img_pth)
        img = cv.imread(op.join(img_pth))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #imshow(img)
        img_set.append(img)
    return img_set


def collect_focal(pth, filename):
    focal_set = []
    focal_set.clear()
    with open(op.join(data_pth, filename), "r") as f:
        for line in f.readlines():
            f = float(line.rstrip('\n'))
            focal_set.append(f)
    return focal_set


def warp_cylindrical_coordinate(img, f):
    # Calculate the coordinate from (x, y) to (theta, h)
    cy_img = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            y_tmp = y - img.shape[0] / 2
            x_tmp = x - img.shape[1] / 2
            y_p = int(round(f * y_tmp / math.sqrt(x_tmp**2 + f**2)) + img.shape[0] / 2)
            x_p = int(round(f * math.atan(x_tmp/f)) + img.shape[1] / 2)
            cy_img[y_p][x_p] = img[y][x]
    return cy_img


def feature_extraction(img):
    # Find out the feature points in our images. (harris corner)
    return


def feature_matching():
    # Match the feature points in two images (RANSAC)
    return


def horizontal_fix():
    # Make sure the whole stitching image is horizontal. (Fix delta y.)
    return


def blending():
    # blend the stitching parts of image
    return


def warp_to_rectangle(img):
    # Assembling panorama.
    # Make the final image beautiful. Crop or Warp.
    return


if __name__ == "__main__":
    # Collect the images to stitch
    data_pth = op.join("images", "parrington")
    img_set = collect_images(data_pth)
    # Collect the focal of image in order 0~N
    focal_filename = "focal.txt"
    focal_set = collect_focal(data_pth, focal_filename)

    cy_img_set = []
    for i, img in enumerate(img_set):
        f = focal_set[i]
        cy_img_set.append(warp_cylindrical_coordinate(img, f))