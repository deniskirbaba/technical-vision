import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('images/lake_forest_rain_small.jpg', cv.IMREAD_GRAYSCALE)

histSize = 256
histRange = (0, 256)
hist = cv.calcHist([img], [0], None, [histSize], histRange)

offset = 75

new_img = np.clip(img + 75, 0, 255)
new_hist = cv.calcHist([new_img], [0], None, [histSize], histRange)

plt.plot(hist, 'b')
plt.plot(new_hist, 'r')
plt.show()

comp = np.concatenate((img, new_img), axis=1)
cv.imshow("images", comp)
cv.waitKey(0)
cv.destroyAllWindows()