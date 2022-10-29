import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_data_folder = "C:/denFiles/git/technical-vision/1lab/images/"

img = cv.imread(img_data_folder + "lake_forest_rain_small.jpg", cv.IMREAD_GRAYSCALE)

max_int = np.max(img)
n_pix = img.size

hist_size = 256
hist_range = (0, 256)

hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
norm_hist = (max_int / n_pix) * hist

cum_hist = np.zeros_like(norm_hist)
for i in range(norm_hist.shape[0]):
    cum_hist[i] = np.sum(norm_hist[:i])

# plt.plot(norm_hist, 'b')
# plt.plot(cum_hist, 'r')
# plt.show()


def new_intensities(prev_int):
    return cum_hist[prev_int]


new_image = np.around(np.squeeze(np.array([new_intensities(i) for i in img]))).astype(np.uint8)

new_hist = cv.calcHist([new_image], [0], None, [hist_size], hist_range)

# plt.plot(hist, 'b')
# plt.plot(new_hist, 'r')
# plt.show()

comp = np.concatenate((img, new_image), axis=1)
cv.imshow("linear alignment", comp)
cv.waitKey(0)
cv.destroyAllWindows()