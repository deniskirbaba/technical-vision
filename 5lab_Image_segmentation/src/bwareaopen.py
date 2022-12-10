import cv2 as cv
import numpy as np

# Remove small objects from binary image
# @param[in] img Input image
# @param[in] dim A minimum size of an area to keep
#
# @param[int] conn Pixel connectivity
# @return An image with components less than dim removed
def bwareaopen(img, dim, conn = 8):
    if img.ndim != 2 or img.dtype != np.uint8:
        return None
    # Find all connected components
    num, labels, stats, centers = cv.connectedComponentsWithStats(img, connectivity=conn)
    # Check size of all connected components
    for i in range(num):
        if stats[i, cv.CC_STAT_AREA] < dim:
            img[labels == i] = 0
    return img