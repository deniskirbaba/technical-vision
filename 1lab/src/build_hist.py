import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img = cv.imread('images/lake_forest_rain.jpg', cv.IMREAD_COLOR)

color = ('b', 'g', 'r')
histSize = 256
histRange = (0, 256)

for i, col in enumerate(color):
    hist = cv.calcHist([img], [i], None, [histSize], histRange)
    plt.plot(hist, color=col)
    plt.xlim(histRange)
plt.show()
