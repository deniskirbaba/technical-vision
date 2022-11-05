import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)

n_rows, n_cols = img.shape[:2]

x_scale = 2.3
y_scale = 1

T = np.float32([[x_scale, 0, 0], [0, y_scale, 0]])

# resized_img = cv.warpAffine(img, T, (int(n_cols * x_scale), int(n_rows * y_scale)))

# using resize func
resized_img = cv.resize(img, (int(n_cols * x_scale), int(n_rows * y_scale)), interpolation=cv.INTER_CUBIC)

cv.imshow("Resised image", resized_img)
cv.imshow("Initial image", img)
cv.waitKey(0)
cv.destroyAllWindows()