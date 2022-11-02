import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(src_path + "snowforest_small_gray.jpg", cv.IMREAD_GRAYSCALE)
i_max = np.max(img)
i_min = np.min(img)

hist_size = 256
hist_range = [0, 256]
hist = cv.calcHist([img], [0], None, [hist_size], hist_range)

alpha = 0.7
lut = np.arange(256, dtype=np.uint8)
lut = (lut - i_min) / (i_max - i_min)
lut = np.where(lut > 0, lut, 0)
lut = np.clip(255 * np.power(lut, alpha), 0, 255).astype(np.uint8)
img_new = cv.LUT(img, lut)
hist_new = cv.calcHist([img_new], [0], None, [hist_size], hist_range)

plt.rcParams["figure.figsize"] = (16,8)
figure, axis = plt.subplots(1, 1)
axis.plot(hist, 'b', label="initial")
axis.plot(hist_new, 'r', label="dynamic range stretching lut")
figure.legend(loc="upper right")
plt.show()

imgs = np.concatenate((img, img_new), axis=1)
cv.imwrite(src_path + "dyn_r_lut.jpg", imgs)
cv.imshow("Dynamic range stretching lut", imgs)
cv.waitKey(0)
cv.destroyAllWindows()