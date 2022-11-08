import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[0:2]

# Set the parameters
x_shift = -100
y_shift = 60

# Form transformation matrix and apply transform
T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
shifted_img = cv.warpAffine(img, T, (n_cols, n_rows))

cv.imwrite(img_path + "shifted_winter_small.jpg", shifted_img)
cv.imshow("Shifting image", np.concatenate((img, shifted_img), axis=1))
cv.waitKey(0)
cv.destroyAllWindows()
