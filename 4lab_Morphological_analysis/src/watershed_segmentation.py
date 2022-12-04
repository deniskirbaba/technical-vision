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
img_path = "C:/denFiles/git/technical-vision/4lab_Morphological_analysis/images/"
img_name = "oranges.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_COLOR)
n_rows, n_cols = img.shape[:2]

# Convert to grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_grayscale.jpg", img_gray)

# Apply threshold
ret_gray, img_threshold = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_threshold.jpg", img_threshold)

# Remove noise
working_img = bwareopen(img_threshold, 40, 4)
el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
working_img = cv.morphologyEx(working_img, cv.MORPH_CLOSE, el, borderType=cv.BORDER_REFLECT)
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_removed_noise.jpg", working_img)

# Find sure foreground
img_fg = cv.distanceTransform(cv.erode(working_img, el, iterations=20), cv.DIST_L2, 5)
cv.normalize(img_fg, img_fg, 0, 255.0, cv.NORM_MINMAX)
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_fg.jpg", np.clip(img_fg, 0, 255).astype(np.uint8))

ret_fg, img_fg = cv.threshold(img_fg, 0.4 * img_fg.max(), 255, cv.THRESH_BINARY)
img_fg = np.clip(img_fg, 0, 255).astype(np.uint8)
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_fg_threshold.jpg", img_fg)

ret, markers = cv.connectedComponents(img_fg)  # Location and markers of sure foreground

# Find sure background
img_bg = cv.dilate(img_threshold, el, iterations=15)
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_bg.jpg", img_bg)

# img_bg = np.zeros_like(working_img)
# markers_bg = markers.copy()
# markers_bg = cv.watershed(img, markers_bg)
# img_bg[markers_bg == -1] = 255
# cv.imwrite(img_path + img_name.rpartition('.')[0] + "_bg.jpg", img_bg)

# Finding unknown region
img_unknown = cv.subtract(img_bg, img_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1
# Now, mark the region of unknown with zero
markers[img_unknown == 255] = 0

# JET colormap areas before watershed segmentation
# The dark blue region shows unknown region. Sure coins are colored with different values.
# Remaining area which are sure background are shown in lighter blue compared to unknown region.
img_jet_before = cv.applyColorMap((markers.astype(np.float32) * 255 / (ret + 1)).astype(np.uint8), colormap=cv.COLORMAP_JET)
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_jet_before.jpg", img_jet_before)

# Do watershed
markers = cv.watershed(img, markers)

# Visualization
res_img = np.copy(img)
res_img[markers == -1] = 255
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_res.jpg", res_img)

# JET colormap areas after watershed segmentation
img_jet_after = cv.applyColorMap((markers.astype(np.float32) * 255 / (ret + 1)).astype(np.uint8), colormap=cv.COLORMAP_JET)
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_jet_after.jpg", img_jet_after)

print("Done")
