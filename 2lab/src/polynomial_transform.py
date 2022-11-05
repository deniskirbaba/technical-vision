import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# set the transform matrix
T = np.float32([[0, 0], [1, 0], [0, 1], [0.00001, 0], [0.002, 0], [0.001, 0]])

polyn_img = np.zeros_like(img)
x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

# Calculate all new X and Y coordinates
x_new = np.round(T[0, 0] + x * T[1, 0] + y * T[2, 0] + x * x * T[3, 0] + x * y * T[4, 0] + y * y * T[5, 0]).astype(np.float32)
y_new = np.round(T[0, 1] + x * T[1, 1] + y * T[2, 1] + x * x * T[3, 1] + x * y * T[4, 1] + y * y * T[5, 1]). astype(np.float32)

# Calculate mask for valid indices
mask = np.logical_and(np.logical_and(x_new >= 0, x_new < n_cols), np.logical_and(y_new >= 0, y_new < n_rows))

# Apply reindexing
polyn_img[y_new[mask].astype(int), x_new[mask].astype(int)] = img[y[mask], x[mask]]

cv.imshow("Polynomial transform", polyn_img)
cv.waitKey(0)
cv.destroyAllWindows()
