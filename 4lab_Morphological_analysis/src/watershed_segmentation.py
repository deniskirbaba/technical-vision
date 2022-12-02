import cv2 as cv
import numpy as np


def bwareopen(img, dim, conn=8):
    if img.ndim > 2:
        return None
    # Find all connected components
    num, labels, stats, centers = cv.connectedComponentsWithStats(img, connectivity=conn)
    # Check size of all connected components
    for i in range(num):
        if stats[i, cv.CC_STAT_AREA] < dim:
            img[labels == i] = 0
    return img


# Reading image
img_path = "C:/denFiles/git/technical-vision/4lab/images/"
img_name = "oranges.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_COLOR)
n_rows, n_cols = img.shape[:2]

# Convert to grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Apply threshold
ret, working_img = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("Threshold", working_img)

# Remove small components
working_img = bwareopen(working_img, 5000, 4)
cv.imshow("Removing small components", working_img)

el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9))
# Apply closing
working_img = cv.morphologyEx(working_img, cv.MORPH_CLOSE, el)
cv.imshow("After closing", working_img)

# Do distance transformation
# Find foreground location
# Define foreground markers
img_fg = cv.distanceTransform(working_img, cv.DIST_L2, 5)

# Normalize the distance image for range = {0.0, 1.0}
# so we can visualize and threshold it
img_fg_norm = np.zeros_like(img_fg)
cv.normalize(img_fg, img_fg_norm, 0, 1.0, cv.NORM_MINMAX)
cv.imshow('Distance Transform Image', img_fg_norm)

ret, img_fg = cv.threshold(img_fg, 0.6 * img_fg.max(), 255, cv.THRESH_BINARY)
img_fg = np.clip(img_fg, 0, 255).astype(np.uint8)  # Convert to uint8
cv.imshow("Thresholded foreground", img_fg)

ret, markers = cv.connectedComponents(img_fg)

# Find background location
img_bg = np.zeros_like(working_img)
markers_bg = markers.copy()
markers_bg = cv.watershed(img, markers_bg)
img_bg[markers_bg == -1] = 255
cv.imshow("Background", img_bg)

# Define undefined area
img_undef = cv.subtract(~img_bg, img_fg)
# Define all markers
markers = markers + 1
markers[img_undef == 255] = 0

# Do watershed
markers = cv.watershed(img, markers)
img[markers == -1] = (0, 0, 255)
cv.imshow("Result", img)

# JET
img_jet = cv.applyColorMap((markers.astype(np.float32) * 255 / (ret + 1)).astype(np.uint8), colormap=cv.COLORMAP_JET)
cv.imshow("JET", img_jet)

cv.waitKey(0)
cv.destroyAllWindows()
