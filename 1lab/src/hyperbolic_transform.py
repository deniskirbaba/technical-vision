import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(src_path + "lenna_small.jpg", cv.IMREAD_GRAYSCALE)

hist_range = [0, 256]
hist_size = 256
alpha = np.min(img / 256)

hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
cumul_hist = np.cumsum(hist) / img.size


def hyperbolic_transform(i_cur):
    return alpha ** cumul_hist[i_cur]


new_img = np.squeeze(np.array([hyperbolic_transform(i) for i in img]))
new_img = np.clip(new_img * 255, 0, 255).astype(np.uint8)
new_hist = cv.calcHist([new_img], [0], None, [hist_size], hist_range)
new_cumul_hist = np.cumsum(new_hist) / new_img.size

plt.plot(cumul_hist, 'b')
plt.plot(new_cumul_hist, 'r')
plt.show()

comp = np.concatenate((img, new_img), axis=1)
cv.imshow("hyperbolic transformation", comp)
cv.waitKey(0)
cv.destroyAllWindows()
