import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Construct the new image array
new_img = img.copy()

# Define the areas
top_left_x, top_left_y = np.meshgrid(np.arange(0, int(n_cols / 2)), np.arange(0, int(n_rows / 2)))
top_right_x, top_right_y = np.meshgrid(np.arange(int(n_cols / 2), n_cols), np.arange(0, int(n_rows / 2)))
bot_left_x, bot_left_y = np.meshgrid(np.arange(0, int(n_cols / 2)), np.arange(int(n_rows / 2), n_rows))
bot_right_x, bot_right_y = np.meshgrid(np.arange(int(n_cols / 2), n_cols), np.arange(int(n_rows / 2), n_rows))

# Stretch by oX the top left part of the image
x_scale = 1.5
T = np.float32([[x_scale, 0, 0], [0, 1, 0]])
new_img[top_left_y, top_left_x] = cv.warpAffine(new_img[top_left_y, top_left_x], T, (top_left_x.shape[1], top_left_x.shape[0]), flags=cv.INTER_CUBIC)

# Stretch by oY the top right part of the image
y_scale = 1.8
T = np.float32([[1, 0, 0], [0, y_scale, 0]])
new_img[top_right_y, top_right_x] = cv.warpAffine(new_img[top_right_y, top_right_x], T, (top_right_x.shape[1], top_right_x.shape[0]), flags=cv.INTER_CUBIC)

# Shift by (-30, 40) bottom left part of the image
x_shift = -30
y_shift = 40
T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
new_img[bot_left_y, bot_left_x] = cv.warpAffine(new_img[bot_left_y, bot_left_x], T, (bot_left_x.shape[1], bot_left_x.shape[0]), flags=cv.INTER_CUBIC)

# Bottom right part leave unchanged

cv.imwrite(img_path + "piecewise_lin_winter.jpg", new_img)
cv.imshow("Piecewise linear transform", new_img)
cv.waitKey(0)
cv.destroyAllWindows()