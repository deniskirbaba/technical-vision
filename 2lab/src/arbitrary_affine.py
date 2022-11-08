import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set points
pts_src = np.float32([[0, 0], [10, 0], [5, 5]])
pts_dst = np.float32([[0.4, 1.4], [7, 3], [4, 6]])

# Get the transformation matrix
T = cv.getAffineTransform(pts_src, pts_dst)

# Apply transformation
new_img = cv.warpAffine(img, T, (n_cols, n_rows))

cv.imwrite(img_path + "arbit_aff_winter.jpg", new_img)
cv.imshow("Arbitrary affine transform", new_img)
cv.waitKey(0)
cv.destroyAllWindows()