from pathlib import Path

import cv2 as cv
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import pdb

def imshow(img, cmap=None):
    plt.imshow(img, cmap)
    plt.show()

def collect_images(path):
    # In cv2, is BGR instead of RGB
    img_set = []
    img_set.clear()
    for img_pth in sorted(Path(data_pth).glob(f'prtn*.jpg')):
        print(img_pth)
        img = cv.imread(op.join(img_pth))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        imshow(img)
        img_set.append(img)
    return img_set


def warp_cylindrical_coordinate(img):
    # Calculate the coordinate from (x, y) to (theta, h)
    return


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

    