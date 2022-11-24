import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka_0.5_impulse_noise.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

# Set parameters
kernel_size = 3

img_new = cv.medianBlur(img, kernel_size)

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(kernel_size) + "_averaging_fil.jpg", img_new)
cv.imshow("Averaging filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()
