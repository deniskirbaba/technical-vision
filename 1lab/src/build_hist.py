import cv2 as cv
from matplotlib import pyplot as plt

img_data_folder = "C:/denFiles/git/technical-vision/1lab/images/"

img = cv.imread(img_data_folder + "lake_forest_rain_small.jpg", cv.IMREAD_COLOR)

color = ('b', 'g', 'r')
hist_size = 256
hist_range = (0, 256)

for i, col in enumerate(color):
    hist = cv.calcHist([img], [i], None, [hist_size], hist_range)
    plt.plot(hist, color=col)
    plt.xlim(hist_range)
plt.show()
