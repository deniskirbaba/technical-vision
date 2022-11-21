import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka_gaus_noise.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

kernel_shape = np.array([5, 5])
img_new = cv.blur(img, kernel_shape)

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(kernel_shape[0]) + 'x' + str(kernel_shape[1]) + "_arith_avg_fil.jpg", img_new)
cv.imshow("Arithmetical averaging filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()