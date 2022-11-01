import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

hist_size = 256
hist_range = (0, 256)

img_folder = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(img_folder + "lake_forest_rain_small.jpg", cv.IMREAD_GRAYSCALE)
hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
plt.plot(hist)

norm_img = img / 255

max_i = np.max(norm_img)
min_i = np.min(norm_img)
non_linearity_factor = 0.5

new_norm_img = np.clip(((norm_img - min_i) / (max_i - min_i)) ** non_linearity_factor, 0, 1)
new_img = (new_norm_img * 255).astype(np.uint8)
new_hist = cv.calcHist([new_img], [0], None, [hist_size], hist_range)
plt.plot(new_hist)
plt.show()

comp = np.concatenate((img, new_img), axis=1)
cv.imshow("images", comp)
cv.waitKey(0)
cv.destroyAllWindows()
