import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set the parameters
x_scale = 0.5
y_scale = 2.2

# Form the transformation matrix
T = np.float32([[x_scale, 0, 0], [0, y_scale, 0]])

# Apply transformation
# resized_img = cv.warpAffine(img, T, (int(n_cols * x_scale), int(n_rows * y_scale)), flags=cv.INTER_CUBIC)

# Using cv.resize func
resized_img = cv.resize(img, (int(n_cols * x_scale), int(n_rows * y_scale)), interpolation=cv.INTER_CUBIC)

cv.imwrite(img_path + "uniform_scale_winter.jpg", resized_img)
cv.imshow("Resised image", resized_img)
cv.imshow("Initial image", img)
cv.waitKey(0)
cv.destroyAllWindows()