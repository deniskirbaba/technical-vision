import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(img_path + "lenna_small.jpg", cv.IMREAD_GRAYSCALE)

i_min = np.min(img) / 255
slope_factor = 2
hist_size = 256
hist_range = (0, 256)

hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
cumul_hist = np.zeros_like(hist)
for i in range(cumul_hist.shape[0]):
    cumul_hist[i] = np.sum(hist[:i])
cumul_hist_norm = cumul_hist / img.size


def exp_transform(cur_i):
    return i_min - (1 / slope_factor) * np.log(1 - cumul_hist_norm[cur_i])


img_new = np.squeeze(np.array([exp_transform(i) for i in img]))
img_new = np.clip(img_new * 255, 0, 255).astype(np.uint8)

new_hist = cv.calcHist([img_new], [0], None, [hist_size], hist_range)
new_cumul_hist = np.zeros_like(new_hist)
for i in range(new_cumul_hist.shape[0]):
    new_cumul_hist[i] = np.sum(new_hist[:i])
new_cumul_hist_norm = new_cumul_hist / img_new.size

plt.plot(cumul_hist_norm, 'b')
plt.plot(new_cumul_hist_norm, 'r')
plt.show()

comp = np.concatenate((img, img_new), axis=1)
cv.imshow("images", comp)
cv.waitKey(0)
cv.destroyAllWindows()
