import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

img_folder = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(img_folder + "snowforest_small_gray.jpg", cv.IMREAD_GRAYSCALE)
i_max = np.max(img)
i_min = np.min(img)

hist_size = 256
hist_range = [0, 256]
hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
cumul_hist = np.cumsum(hist)

non_linearity_factor = 0.7
new_img = np.power((img.astype(np.float32) - i_min) / (i_max - i_min), non_linearity_factor)
new_img = (255 * new_img).astype(np.uint8)
new_hist = cv.calcHist([new_img], [0], None, [hist_size], hist_range)
new_cumul_hist = np.cumsum(new_hist)

plt.rcParams["figure.figsize"] = (16,8)
figure, axis = plt.subplots(1, 2)
axis[0].plot(cumul_hist, 'b', label="Initial")
axis[0].plot(new_cumul_hist, 'r', label="Stretching dynamic range")
axis[1].plot(hist, 'b')
axis[1].plot(new_hist, 'r')
figure.legend(loc="upper right")
plt.show()

imgs = np.concatenate((img, new_img), axis=1)
cv.imwrite(img_folder + "stretch_dyn_r.jpg", imgs)
cv.imshow("Stretching dynamic range", imgs)
cv.waitKey(0)
cv.destroyAllWindows()
