import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img = cv.imread(img_path + "oulanka.jpg", cv.IMREAD_GRAYSCALE)

# Variance parameter
var = 0.05
# Generate random numbers
rng = np.random.default_rng()
gauss = rng.normal(0, var ** 0.5, img.shape)
# Adding noise
img_new = np.clip(img + img * gauss, 0, 255).astype(np.uint8)

cv.imwrite(img_path + "oulanka_mul_noise.jpg", img_new)
cv.imshow("Multiplicative noise", img_new)
cv.waitKey(0)
cv.destroyAllWindows()