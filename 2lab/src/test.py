import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "rowan.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Form x,y grid
x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

x_new_custom = x ** 2
y_new_custom = y

x_new_remap = x ** 0.5
y_new_remap = y

mask = np.logical_and(np.logical_and(x_new_custom >= 0, x_new_custom < n_cols), np.logical_and(y_new_custom >= 0, y_new_custom < n_rows))

new_img_custom = np.zeros_like(img)
new_img_custom[y_new_custom[mask].astype(int), x_new_custom[mask].astype(int)] = img[y[mask], x[mask]]

new_img_remap = cv.remap(img, x_new_remap.astype(np.float32), y_new_remap.astype(np.float32), interpolation=None)

cv.imshow("manual", new_img_custom)
cv.imshow("remap", new_img_remap)
cv.waitKey(0)
cv.destroyAllWindows()