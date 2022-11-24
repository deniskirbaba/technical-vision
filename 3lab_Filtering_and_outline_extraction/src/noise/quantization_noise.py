import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

# Convert to float and normalize
img_float = img.astype(np.float32) / 255

# Generate random numbers
rng = np.random.default_rng()

# Find distribution parameter lambda
labmda = len(np.unique(img_float))
labmda = 2 ** np.ceil(np.log2(labmda))
# Add noise
img_new = (255 * ((rng.poisson(img_float * labmda) / float(labmda)).clip(0, 1))).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + "_quan_noise.jpg", img_new)
cv.imshow("Quantization noise", img_new)
cv.waitKey(0)
cv.destroyAllWindows()