import cv2 as cv
import numpy as np

# Implementation of MATLAB's imfill(I, 'holes') function
# @param[in] img Image to process
# @return An image with holes removed
def imfill(img):
    if img.ndim != 2 or img.dtype != np.uint8:
        return None
    rows, cols = img.shape[0:2]
    mask = img.copy()
    # Fill mask from all horizontal borders
    for i in range(cols):
        if mask[0, i] == 0:
            cv.floodFill(mask, None, (i, 0), 255, 10, 10)
        if mask[rows - 1, i] == 0:
            cv.floodFill(mask, None, (i, rows - 1), 255, 10, 10)
    # Fill mask from all vertical borders
    for i in range(rows):
        if mask[i, 0] == 0:
            cv.floodFill(mask, None, (0, i), 255, 10, 10)
        if mask[i, cols - 1] == 0:
            cv.floodFill(mask, None, (cols - 1, i), 255, 10, 10)
    # Use the mask to create a resulting image
    res_img = img.copy()
    res_img[mask == 0] = 255
    return res_img