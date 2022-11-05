import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# stretch the right side of the image
x_scale = 2
T = np.float32([[x_scale, 0, 0], [0, 1, 0]])
piecewise_lin_img = img.copy()
piecewise_lin_img[:, int(n_cols / 2):] = cv.warpAffine(piecewise_lin_img[:, int(n_cols / 2):], T, (n_cols - int(n_cols / 2), n_rows))

cv.imshow("Piecewise linear transform", piecewise_lin_img)
cv.waitKey(0)
cv.destroyAllWindows()