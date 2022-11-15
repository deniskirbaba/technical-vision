import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

# Set parameters
var = 0.05
mean = 0
# Generate random numbers
rng = np.random.default_rng()
gauss = rng.normal(mean, var ** 0.5, img.shape)
# Adding noise
img_new = np.clip(img + gauss * 255, 0, 255).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + "_gaus_noise.jpg", img_new)
cv.imshow("Gaussian noise", img_new)
cv.waitKey(0)
cv.destroyAllWindows()