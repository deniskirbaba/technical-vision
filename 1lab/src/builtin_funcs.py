import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(src_path + "lenna_small.jpg", cv.IMREAD_GRAYSCALE)
hist_size = 256
hist_range = [0, 256]

hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
eq_img = cv.equalizeHist(img)
eq_hist = cv.calcHist([eq_img], [0], None, [hist_size], hist_range)

clahe = cv.createCLAHE()
clahe_img = cv.CLAHE.apply(clahe, img)
clahe_hist = cv.calcHist([clahe_img], [0], None, [hist_size], hist_range)

plt.plot(hist, 'r')
plt.plot(eq_hist, 'b')
plt.plot(clahe_hist, 'g')
plt.show()

imgs = np.concatenate((img, eq_img, clahe_img), axis=1)
cv.imshow("built-in funcs", imgs)
cv.waitKey(0)
cv.destroyAllWindows()
