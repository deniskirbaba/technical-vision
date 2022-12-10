import cv2 as cv
import numpy as np
import skimage
from bwareaopen import bwareaopen
from imfill import imfill

# Reading an image
img_path = "../images/"
img_name = "satellite_1.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_COLOR)

# Convert to grayscale
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale image", img_gray)

# Calculate entropy and then normalize
ent = skimage.filters.rank.entropy(img_gray, skimage.morphology.disk(13)).astype(np.float32)
ent_norm = (ent - np.min(ent)) / (np.max(ent) - np.min(ent))
cv.imshow("Entropy", np.uint8(ent_norm * 255))

# Binarize the resulting normalized array using the Otsu method
ret, bin_ent = cv.threshold(np.uint8(ent_norm * 255), 0, 255, cv.THRESH_OTSU)
# cv.imshow("Binary entropy", bin_ent)

# Morphological filtering
# 1. Remove small objects from binary image
bin_ent_bwao = bwareaopen(bin_ent, 2000, conn=8)
# cv.imshow("Bin ent bwao", bin_ent_bwao)
# 2. Remove internal defects with closing operation
el = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
cl_bin_ent_bwao = cv.morphologyEx(bin_ent_bwao, cv.MORPH_CLOSE, el)
# cv.imshow("Close bin ent bwao", cl_bin_ent_bwao)
# 3. Fill remaining large holes
fil_cl_bin_ent_bwao = imfill(cl_bin_ent_bwao)
# cv.imshow("Filled close bin ent bwao", fil_cl_bin_ent_bwao)

# Find shape contours and then draw them on a black background
contours, h = cv.findContours(fil_cl_bin_ent_bwao, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
boundary = np.zeros_like(img_gray)
cv.drawContours(boundary, contours, -1, 255, 1)
# cv.imshow("Contour image", boundary)

# Apply the found border to the source image
segment_results = img.copy()
segment_results[boundary != 0] = 255
# cv.imshow("Contour image + source image", segment_results)

# Excluding the water area from the source image and do the same steps to select the
# remaining texture of the land
img_2 = img_gray.copy()
img_2[contours == 0] = 0
# Entropy and binarization
ent_2 = skimage.filters.rank.entropy(img_2, skimage.morphology.square(9)).astype(np.float32)
ent_norm_2 = (ent_2 - ent_2.min()) / (ent_2.max() - ent_2.min())
ret_2, bin_ent_2 = cv.threshold(np.uint8(ent_norm_2 * 255), 0, 255, cv.THRESH_OTSU)
# Filtering
bin_ent_bwao_2 = bwareaopen(bin_ent_2, 2000, conn=8)
cl_bin_ent_bwao_2 = cv.morphologyEx(bin_ent_bwao_2, cv.MORPH_CLOSE, el)
fil_cl_bin_ent_bwao_2 = imfill(cl_bin_ent_bwao_2)
# Select boundary
contours_2, h_2 = cv.findContours(fil_cl_bin_ent_bwao_2, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
boundary_2 = np.zeros_like(img_2)
cv.drawContours(boundary_2, contours_2, -1, 255, 1)
segment_results_2 = img.copy()
segment_results_2[boundary_2 != 0] = 255

# Select textures of water and land basing on either of masks
texture_1 = img_gray.copy()
texture_1[cl_bin_ent_bwao_2 == 0] = 0
texture_2 = img_gray.copy()
texture_2[cl_bin_ent_bwao_2 != 0] = 0

# Show textures
cv.imshow("Texture 1", texture_1)
cv.imshow("Texture 2", texture_2)
cv.waitKey(0)
cv.destroyAllWindows()

print("End of the program...")
