import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(img_path + "spb.jpg", cv.IMREAD_GRAYSCALE)
i_min = np.min(img)

hist_size = 256
hist_range = (0, 256)
hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
cumul_hist = np.cumsum(hist) / img.size
cumul_hist = np.where(cumul_hist < 0.99999, cumul_hist, 0.99999)

slope_factor = 2
new_img = np.clip(i_min - (1 / slope_factor) * 255 * np.log(1 - cumul_hist[img]), 0, 255).astype(np.uint8)
new_hist = cv.calcHist([new_img], [0], None, [hist_size], hist_range)
new_cumul_hist = np.cumsum(new_hist) / new_img.size

plt.rcParams["figure.figsize"] = (16,8)
figure, axis = plt.subplots(1, 2)
axis[0].plot(cumul_hist, 'b', label="Initial")
axis[0].plot(new_cumul_hist, 'r', label="Exp transformation")
axis[1].plot(hist, 'b')
axis[1].plot(new_hist, 'r')
figure.legend(loc="upper right")
plt.show()

imgs = np.concatenate((img, new_img), axis=1)
cv.imwrite(img_path + "exp_transform.jpg", imgs)
cv.imshow("Exp transformation", imgs)
cv.waitKey(0)
cv.destroyAllWindows()
