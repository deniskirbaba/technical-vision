import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "kitchen.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Form x,y grid
x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

# Shift and normalize grid
x_mid = n_cols / 2
y_mid = n_rows / 2
x = (x - x_mid) / x_mid
y = (y - y_mid) / y_mid

# Set linear expansion factor
b_0 = 1.005

# Convert to polar and do transformation
r, theta = cv.cartToPolar(x, y)
F3 = 0.5
r = b_0 * r + F3 * r ** 3

# Undo conversion, normalization and shift
u, v = cv.polarToCart(r, theta)
u = u * x_mid + x_mid
v = v * y_mid + y_mid

# Calculate mask for valid indices
mask = np.logical_and(np.logical_and(u >= 0, u < n_cols), np.logical_and(v >= 0, v < n_rows))

# Undo normalization and shift
x = x * x_mid + x_mid
y = y * y_mid + y_mid

# Apply reindexing
dist_img = np.zeros_like(img)
dist_img[v[mask].astype(int), u[mask].astype(int)] = img[y[mask].astype(int), x[mask].astype(int)]

# Find undefined pixels of new image
undef = np.zeros_like(img)
undef[v[mask].astype(int), u[mask].astype(int)] = 255  # Undefined pixels corresponds to the black

# Apply interpolation
undef_arr = np.argwhere(undef == 0)  # An array containing undefined pixels
for undef_p in undef_arr:
    intensities = []
    # Construct 3x3 squared punctured area
    area = np.array([[undef_p[0] - 1, undef_p[1] - 1], [undef_p[0] - 1, undef_p[1]], [undef_p[0] - 1, undef_p[1] + 1],
                     [undef_p[0], undef_p[1] - 1], [undef_p[0], undef_p[1] + 1],
                     [undef_p[0] + 1, undef_p[1] - 1], [undef_p[0] + 1, undef_p[1]], [undef_p[0] + 1, undef_p[1] + 1]])
    # Find the area average intensities of the defined pixels
    for p in area:
        if (p[0] >= 0) and (p[0] < n_rows) and (p[1] >= 0) and (p[1] < n_cols) and (undef[p[0], p[1]] == 255):
            intensities.append(dist_img[p[0], p[1]])
    dist_img[undef_p[0], undef_p[1]] = int(np.mean(intensities))  # Assign the found intensity

cv.imwrite(img_path + "kitchen_distortion_custom.jpg", dist_img)
cv.imshow("Distortion custom", dist_img)
cv.waitKey(0)
cv.destroyAllWindows()