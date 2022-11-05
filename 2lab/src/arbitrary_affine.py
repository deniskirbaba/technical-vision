import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# get the transform matrix
pts_src = np.float32([[50, 300], [150, 200], [50, 50]])
pts_dst = np.float32([[50, 200], [250, 200], [50, 100]])
T = cv.getAffineTransform(pts_src, pts_dst)

new_img = cv.warpAffine(img, T, (n_cols, n_rows))

cv.imshow("Arbitrary affine transform", new_img)
cv.waitKey(0)
cv.destroyAllWindows()