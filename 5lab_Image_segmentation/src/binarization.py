import cv2 as cv
import numpy as np

# Reading image
img_path = "C:/denFiles/git/technical-vision/5lab_Image_segmentation/images/"
img_name = "house.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Finding threshold value
# 1. Arithmetic mean of max and min intensities
# t = (np.max(img) - np.min(img)) / 2

# 2. Based on brightness gradient
# mx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
# my = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
# # Compute gradients
# grad_x = cv.filter2D(img, -1, mx, borderType=cv.BORDER_REFLECT)
# grad_y = cv.filter2D(img, -1, my, borderType=cv.BORDER_REFLECT)
# # Find modulus
# grad = np.maximum(np.abs(grad_x), np.abs(grad_y))
# # Compute t
# t = np.sum(np.multiply(grad, img)) / np.sum(grad)

# 3. Otsu binarization
# ret, bin_img = cv.threshold(img, 0, np.max(img), cv.THRESH_OTSU)

# 4. Adaptive method
bin_img = cv.adaptiveThreshold(img, np.max(img), cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

# Binarization with one t
# t = 127
# ret, bin_img = cv.threshold(img, t, 255, cv.THRESH_BINARY)

# Double binarization
# t1 = 127
# t2 = 200
# ret, bin_img = cv.threshold(img, t2, 255, cv.THRESH_TOZERO_INV)
# ret, bin_img = cv.threshold(bin_img, t1, 255, cv.THRESH_BINARY)

# Show image
# cv.imwrite(img_path + img_name.rpartition('.')[0] + "_black_bg.jpg", img)
cv.imshow("Result", bin_img)
cv.waitKey(0)
cv.destroyAllWindows()

print("Done")
