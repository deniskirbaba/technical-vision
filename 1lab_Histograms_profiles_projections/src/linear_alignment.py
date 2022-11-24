import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

data_folder = "C:/denFiles/git/technical-vision/1lab/images/"

# reading an image
img = cv.imread(data_folder + "spb.jpg", cv.IMREAD_GRAYSCALE)

l = 255 # max intensity
n_pix = img.size

# calculate cumulative histogram for the initial image
hist_size = 256
hist_range = (0, 256)
hist = cv.calcHist([img], [0], None, [hist_size], hist_range)
norm_hist = (l / n_pix) * hist
cumul_hist = np.clip(np.cumsum(norm_hist), 0, 255)

# get the linear aligned image
new_img = cumul_hist[img].astype(np.uint8)
new_hist = cv.calcHist([new_img], [0], None, [hist_size], hist_range)
new_norm_hist = (l / n_pix) * new_hist
new_cumul_hist = np.clip(np.cumsum(new_norm_hist), 0, 255)

plt.rcParams["figure.figsize"] = (16,8)
figure, axis = plt.subplots(1, 2)
axis[0].plot(cumul_hist, 'b', label="Initial")
axis[0].plot(new_cumul_hist, 'r', label="Linear alignment")
axis[1].plot(hist, 'b')
axis[1].plot(new_hist, 'r')
figure.legend(loc="upper right")
plt.show()

imgs = np.concatenate((img, new_img), axis=1)
cv.imwrite(data_folder + "lin_al.jpg", imgs)
cv.imshow("Linear alignment", imgs)
cv.waitKey(0)
cv.destroyAllWindows()