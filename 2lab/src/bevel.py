import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

bevel_x = 0.1
bevel_y = 0

T = np.float32([[1, bevel_x, 0], [bevel_y, 1, 0]])

bevelled_img = cv.warpAffine(img, T, (n_cols, n_rows))

cv.imshow("Bevelled image", bevelled_img)
cv.waitKey(0)
cv.destroyAllWindows()