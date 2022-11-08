import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

xi, yi = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

# Shift and normalize grid
xmid = n_cols / 2
ymid = n_rows / 2
xi = (xi - xmid) / xmid
yi = (yi - ymid) / ymid

# Convert to polar and do transformation
r, theta = cv.cartToPolar(xi, yi)
F3 = -0.5
F5 = 0
r = r + F3 * r ** 3 + F5 * r ** 5

# Undo conversion, normalization and shift
u, v = cv.polarToCart(r, theta)
u = u * xmid + xmid
v = v * ymid + ymid

# Do remapping
new_img = cv.remap(img, u.astype(np.float32), v.astype(np.float32), cv.INTER_LINEAR)

cv.imshow("Distortion", new_img)
cv.waitKey(0)
cv.destroyAllWindows()