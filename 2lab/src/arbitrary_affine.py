import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "guitar_left_gray.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set points
pts_src = np.float32([[765, 471], [871, 252], [888, 1152]])
pts_dst = np.float32([[65, 312], [104, 54], [159, 1000]])

# Get the transformation matrix
T = cv.getAffineTransform(pts_src, pts_dst)

# Apply transformation
new_img = cv.warpAffine(img, T, (n_cols, n_rows))

# cv.imwrite(img_path + "arbit_aff_winter.jpg", new_img)
cv.imshow("Arbitrary affine transform", new_img)
cv.waitKey(0)
cv.destroyAllWindows()