import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

# Generate random numbers
rng = np.random.default_rng()
uniform = rng.uniform(low=0, high=1, size=img.shape)
# Adding noise
img_new = np.clip(img * uniform, 0, 255).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + "_mult_noise.jpg", img_new)
cv.imshow("Multiplicative noise", img_new)
cv.waitKey(0)
cv.destroyAllWindows()