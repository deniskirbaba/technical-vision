import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka_quan_noise.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

# Set parameters
kernel_shape = np.array([3, 3])
sigmaX = 5

# Apply filtering
img_new = cv.GaussianBlur(img, kernel_shape, sigmaX, cv.BORDER_REFLECT)

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(kernel_shape[0]) + 'x' + str(kernel_shape[1]) + '_' + str(sigmaX) + "_gauss_fil.jpg", img_new)
cv.imshow("Gaussian filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()
