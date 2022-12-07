import cv2 as cv
import numpy as np
from weighted_median_filtering import weighted_median_filtering

div_state = np.seterr(divide='ignore')

# Reading image
img_path = "C:/denFiles/git/technical-vision/5lab_Image_segmentation/images/"
img_name = "face7.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_COLOR)
n_rows, n_cols = img.shape[:2]

# 1. Uniform daylight illumination
# mask = np.logical_and(np.logical_and(img[:, :, 0] > 20, img[:, :, 1] > 40), img[:, :, 2] > 95)  # R>95, G>40, B>20 conditions
# mask = np.logical_and(np.logical_and(mask, img[:, :, 2] > img[:, :, 1]), img[:, :, 2] > img[:, :, 0])  # R>G, R>B conditions
# mask = np.logical_and(mask, np.abs(img[:, :, 2] - img[:, :, 1]) > 15)  # |R-G|>15 condition
# mask = np.logical_and(mask, (np.max(img, axis=2) - np.min(img, axis=2)) > 15)  # maxRGB - minRGB > 15 condition

# 2. Under flashlight or daylight lateral illumination
# mask = np.logical_and(np.logical_and(img[:, :, 0] > 170, img[:, :, 1] > 210), img[:, :, 2] > 220)  # R>220, G>210, B>170 conditions
# mask = np.logical_and(np.logical_and(mask, img[:, :, 1] > img[:, :, 0]), img[:, :, 2] > img[:, :, 0])  # G>B, R>B conditions
# mask = np.logical_and(mask, np.abs(img[:, :, 2] - img[:, :, 1]) <= 15)  # |R-G|<=15 condition

# 3. Using normalized RGB values
# Convert to initial img to float
img = img.astype(np.float64)
# Convert to normalized RGB
bgr_sum_img = np.sum(img, axis=2)
img_norm_b = np.divide(img[:, :, 0], bgr_sum_img)
img_norm_g = np.divide(img[:, :, 1], bgr_sum_img)
img_norm_r = np.divide(img[:, :, 2], bgr_sum_img)

# Create mask
denum = np.power(img_norm_r + img_norm_g + img_norm_b, 2)  # Auxiliary array
mask = np.logical_and(np.divide(np.multiply(img_norm_r, img_norm_b), denum) > 0.107, np.divide(np.multiply(img_norm_r, img_norm_g), denum) > 0.112)
mask = np.logical_and(mask, np.divide(img_norm_r, img_norm_g) > 1.185)

# Apply mask
new_img = np.zeros_like(mask, dtype=np.uint8)
new_img[mask] = 255

# Apply median filtering
new_img = weighted_median_filtering(new_img, 3)

# Show result
cv.imshow("Result", new_img)
cv.waitKey(0)
cv.destroyAllWindows()

print("Done")
