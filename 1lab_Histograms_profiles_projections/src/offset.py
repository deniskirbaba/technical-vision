import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_data_folder = "C:/denFiles/git/technical-vision/1lab/images/"

# reading image
img = cv.imread(img_data_folder + "lake_forest_rain_small.jpg", cv.IMREAD_GRAYSCALE)

hist_size = 256
hist_range = (0, 256)
hist = cv.calcHist([img], [0], None, [hist_size], hist_range)

offset = 75

# make a shift
new_img = np.clip(img.astype(np.int16) + offset, 0, 255).astype(np.uint8)
new_hist = cv.calcHist([new_img], [0], None, [hist_size], hist_range)

plt.plot(hist, 'b')
plt.plot(new_hist, 'r')
plt.show()

comp = np.concatenate((img, new_img), axis=1)
cv.imshow("images", comp)
cv.waitKey(0)
cv.destroyAllWindows()