import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
# reading an image
img = cv.imread(src_path + "snowforest_small_gray.jpg", cv.IMREAD_GRAYSCALE)

hist_size = 256
hist_range = [0, 256]
hist = cv.calcHist([img], [0], None, [hist_size], hist_range)

# use equalizeHist()
eq_img = cv.equalizeHist(img)
eq_hist = cv.calcHist([eq_img], [0], None, [hist_size], hist_range)

# ust CLAHE
clahe = cv.createCLAHE()
clahe_img = cv.CLAHE.apply(clahe, img)
clahe_hist = cv.calcHist([clahe_img], [0], None, [hist_size], hist_range)

plt.rcParams["figure.figsize"] = (16,8)
plt.plot(hist, 'r', label="initial")
plt.plot(eq_hist, 'b', label="equalised")
plt.plot(clahe_hist, 'g', label="CLAHE")
plt.legend()
plt.show()

imgs = np.concatenate((img, eq_img, clahe_img), axis=1)
cv.imshow("built-in funcs", imgs)
cv.imwrite(src_path + "builtin_funcs.jpg", imgs)
cv.waitKey(0)
cv.destroyAllWindows()
