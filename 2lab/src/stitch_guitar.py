import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img_left = cv.imread(img_path + "guitar_left_gray.jpg", cv.IMREAD_GRAYSCALE)
img_right = cv.imread(img_path + "guitar_right_gray.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img_left.shape[:2]

# Corresponding dots
a_right = np.array([104, 54])
b_right = np.array([99, 860])
c_right = np.array([159, 1000])
a_left = np.array([871, 252])
b_left = np.array([819, 993])
c_left = np.array([888, 1152])

# Find coeffs of the affine transform
# To translate the left-hand image into the right-hand coordinate system
x_coeffs = np.matmul(np.linalg.inv([[1, a_left[0], a_left[1]], [1, b_left[0], b_left[1]], [1, c_left[0], c_left[1]]]), np.array([[a_right[0]], [b_right[0]], [c_right[0]]]))
y_coeffs = np.matmul(np.linalg.inv([[1, a_left[0], a_left[1]], [1, b_left[0], b_left[1]], [1, c_left[0], c_left[1]]]), np.array([[a_right[1]], [b_right[1]], [c_right[1]]]))

# Create grid corresponding to our pixels
x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

# Calculate all new X and Y coordinates
x_new = np.round( x_coeffs[1,0] * x + x_coeffs[2,0] * y).astype(np.float32)
y_new = np.round(y_coeffs[0,0] + y_coeffs[1,0] * x + y_coeffs[2,0] * y).astype(np.float32)

# Calculate mask for valid indices
mask = np.logical_and(np.logical_and(x_new >= 0, x_new < n_cols), np.logical_and(y_new >= 0, y_new < n_rows))

# Apply reindexing
new_left_img = np.zeros_like(img_left)
new_left_img[y_new[mask].astype(int), x_new[mask].astype(int)] = img_left[y[mask], x[mask]]

cv.imshow("New left guitar image", new_left_img)
cv.waitKey(0)
cv.destroyAllWindows()
