import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img = cv.imread(img_path + "oulanka.jpg", cv.IMREAD_GRAYSCALE)

# Generate random numbers
rng = np.random.default_rng()
poisson = rng.poisson(100, img.shape)
# Adding noise
img_new = np.clip(img + poisson, 0, 255).astype(np.uint8)

cv.imwrite(img_path + "oulanka_add_noise.jpg", img_new)
cv.imshow("Additive noise", img_new)
cv.waitKey(0)
cv.destroyAllWindows()