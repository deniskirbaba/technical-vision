import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img = cv.imread(img_path + "oulanka_add_noise.jpg", cv.IMREAD_GRAYSCALE)

kernel = np.float64([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9
img_new = cv.filter2D(img, -1, kernel, borderType=cv.BORDER_REFLECT)

cv.imshow("Local sliding window filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()