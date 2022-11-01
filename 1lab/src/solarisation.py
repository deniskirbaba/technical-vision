import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(src_path + "lenna_small.jpg", cv.IMREAD_GRAYSCALE).astype(np.float32)

hist_range = [0, 256]
hist_size = 256
dynamic_range_factor = 4
i_max = np.max(img)

hist = cv.calcHist([img], [0], None, [hist_size], hist_range)


def solarization(cur_i):
    return dynamic_range_factor * img[cur_i] * (i_max - img[cur_i]) / i_max


# new_img = np.squeeze(np.array([solarization(i) for i in img]))
new_img = dynamic_range_factor * img * (i_max - img) / i_max
new_img = np.clip(new_img, 0, 255).astype(np.uint8)
new_hist = cv.calcHist([new_img], [0], None, [hist_size], hist_range)

plt.plot(hist, 'b')
plt.plot(new_hist, 'r')
plt.show()

comp = np.concatenate((img.astype(np.uint8), new_img.astype(np.uint8)), axis=1)
cv.imshow("solarization", comp)
cv.waitKey(0)
cv.destroyAllWindows()
