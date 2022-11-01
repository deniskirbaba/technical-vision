import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(src_path + "lenna_small.jpg", cv.IMREAD_GRAYSCALE)

i_max = np.max(img) / 255
i_min = np.min(img) / 255
hist_range = [0, 256]
hist_size = 256

hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
cumul_hist = np.zeros_like(hist)
for i in range(cumul_hist.shape[0]):
    cumul_hist[i] = np.sum(hist[:i])
cumul_hist = cumul_hist / img.size


def uniform_transform(i_cur):
    return (i_max - i_min) * cumul_hist[i_cur] + i_min


new_img = np.squeeze(np.array([uniform_transform(i) for i in img]))
new_img = np.clip(new_img * 255, 0, 255).astype(np.uint8)
new_hist = cv.calcHist([new_img], [0], None, [hist_size], hist_range)
new_cumul_hist = np.zeros_like(new_hist)
for i in range(new_cumul_hist.shape[0]):
    new_cumul_hist[i] = np.sum(new_hist[:i])
new_cumul_hist = new_cumul_hist / new_img.size

plt.plot(hist, 'b')
plt.plot(new_hist, 'r')
plt.show()

comp = np.concatenate((img, new_img), axis=1)
cv.imshow("uniform transformation", comp)
cv.waitKey(0)
cv.destroyAllWindows()