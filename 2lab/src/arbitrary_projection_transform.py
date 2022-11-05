import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# set the transform matrix
# T = np.float32([[1.1, 0.2, 0.00075], [0.35, 1.1, 0.0005], [0, 0, 1]])

# or using builtin func
pts_src = np.float32([[50, 461], [461, 461], [461, 50], [50, 50]])
pts_dst = np.float32([[50, 461], [461, 440], [450, 10], [100, 50]])
T = cv.getPerspectiveTransform(pts_src, pts_dst)

new_img = cv.warpPerspective(img, T, (n_cols, n_rows))

cv.imshow("Arbitrary projective transform", new_img)
cv.waitKey(0)
cv.destroyAllWindows()