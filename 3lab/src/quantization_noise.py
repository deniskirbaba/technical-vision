import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

rng = np.random.default_rng()
img_float = img.astype(np.float32) / 255
vals = len(np.unique(img_float))
vals = 2 ** np.ceil(np.log2(vals))
img_new = (255 * ((rng.poisson(img_float * vals) / float(vals)).clip(0, 1))).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + "_quan_noise.jpg", img_new)
cv.imshow("Quantization noise", img_new)
cv.waitKey(0)
cv.destroyAllWindows()