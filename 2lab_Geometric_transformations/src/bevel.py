import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set the parameters
s_x = 0.3
s_y = -0.1

# Make the transformation matrix
T = np.float32([[1, s_x, 0], [s_y, 1, 0]])

# Apply transformation
bevelled_img = cv.warpAffine(img, T, (n_cols, n_rows), flags=cv.INTER_CUBIC)

cv.imwrite(img_path + "bevelled_winter.jpg", bevelled_img)
cv.imshow("Bevelled image", bevelled_img)
cv.waitKey(0)
cv.destroyAllWindows()