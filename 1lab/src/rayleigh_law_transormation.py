import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(src_path + "snowforest_small_gray.jpg", cv.IMREAD_GRAYSCALE)
i_min = np.min(img)

hist_range = [0, 256]
hist_size = 256
hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
cumul_hist = np.cumsum(hist) / img.size
cumul_hist = np.where(cumul_hist < 0.99999, cumul_hist, 0.99999)

alpha = 0.3
new_img = i_min + 255 * (2 * (alpha ** 2) * np.log(1/(1 - cumul_hist[img]))) ** (1 / 2)
new_img = np.clip(new_img, 0, 255).astype(np.uint8)
new_hist = cv.calcHist([new_img], [0], None, [hist_size], hist_range)
new_cumul_hist = np.cumsum(new_hist) / new_img.size

plt.rcParams["figure.figsize"] = (16,8)
figure, axis = plt.subplots(1, 2)
axis[0].plot(cumul_hist, 'b', label="Initial")
axis[0].plot(new_cumul_hist, 'r', label="Rayleigh transformation")
axis[1].plot(hist, 'b')
axis[1].plot(new_hist, 'r')
figure.legend(loc="upper right")
plt.show()

imgs = np.concatenate((img, new_img), axis=1)
cv.imwrite(src_path + "rayleigh_transform.jpg", imgs)
cv.imshow("Rayleigh transformation", imgs)
cv.waitKey(0)
cv.destroyAllWindows()
