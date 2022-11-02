import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(src_path + "lenna_small.jpg", cv.IMREAD_GRAYSCALE)
hist_size = 256
hist_range = [0, 256]

i_max = np.max(img)
i_min = np.min(img)

hist = cv.calcHist([img], [0], None, [hist_size], hist_range)

alpha = 0.5
lut = np.arange(256, dtype=np.uint8)
lut = (lut - i_min) / (i_max - i_min)
lut = np.where(lut > 0, lut, 0)
lut = np.clip(255 * np.power(lut, alpha), 0, 255).astype(np.uint8)
img_new = cv.LUT(img, lut)
hist_new = cv.calcHist([img_new], [0], None, [hist_size], hist_range)

plt.plot(hist, 'r')
plt.plot(hist_new, 'b')
plt.show()

imgs = np.concatenate((img, img_new), axis=1)
cv.imshow("lut transform", imgs)
cv.waitKey(0)
cv.destroyAllWindows()