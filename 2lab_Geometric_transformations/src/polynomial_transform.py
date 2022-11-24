import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set the transformation matrix
T = np.float32([[0, 0], [0.7, 0], [0, 0.9], [0.00001, 0], [0.002, 0], [0.001, 0]])

# Create grid corresponding to our pixels
x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

# Calculate all new X and Y coordinates
x_new = np.round(T[0, 0] + x * T[1, 0] + y * T[2, 0] + x * x * T[3, 0] + x * y * T[4, 0] + y * y * T[5, 0]).astype(
    np.float32)
y_new = np.round(T[0, 1] + x * T[1, 1] + y * T[2, 1] + x * x * T[3, 1] + x * y * T[4, 1] + y * y * T[5, 1]).astype(
    np.float32)

# Calculate mask for valid indices
mask = np.logical_and(np.logical_and(x_new >= 0, x_new < n_cols), np.logical_and(y_new >= 0, y_new < n_rows))

# Apply reindexing
polyn_img = np.zeros_like(img)
polyn_img[y_new[mask].astype(int), x_new[mask].astype(int)] = img[y[mask], x[mask]]

# Find undefined pixels and borders of new image
undef_all = np.zeros_like(img)
borders = np.zeros_like(img)
undef_inner = np.zeros_like(img)
undef_all[y_new[mask].astype(int), x_new[mask].astype(int)] = 255  # Undefined pixels corresponds to the black
for i in range(n_rows):
    def_pixels = np.argwhere(undef_all[i, :] == 255)
    if def_pixels.size > 2:
        img_start = np.squeeze(def_pixels[0])
        img_end = np.squeeze(def_pixels[-1])
        borders[i, [img_start, img_end]] = 255
        row_undef_pix = np.argwhere(undef_all[i, img_start:img_end] == 0)
        undef_inner[i, row_undef_pix + img_start] = 255

# Apply interpolation
undef_inner_arr = np.argwhere(undef_inner == 255)  # An array containing inner undefined pixels
for undef_p in undef_inner_arr:
    intensities = []
    # Construct 5x5 punctured area
    area = np.array([[undef_p[0] - 2, undef_p[1] - 2], [undef_p[0] - 2, undef_p[1] - 1], [undef_p[0] - 2, undef_p[1]], [undef_p[0] - 2, undef_p[1] + 1], [undef_p[0] - 2, undef_p[1] + 2],
                     [undef_p[0] - 1, undef_p[1] - 2], [undef_p[0] - 1, undef_p[1] - 1], [undef_p[0] - 1, undef_p[1]], [undef_p[0] - 1, undef_p[1] + 1], [undef_p[0] - 1, undef_p[1] + 2],
                     [undef_p[0], undef_p[1] - 2], [undef_p[0], undef_p[1] - 1], [undef_p[0], undef_p[1] + 1], [undef_p[0], undef_p[1] + 2],
                     [undef_p[0] + 1, undef_p[1] - 2], [undef_p[0] + 1, undef_p[1] - 1], [undef_p[0] + 1, undef_p[1]], [undef_p[0] + 1, undef_p[1] + 1], [undef_p[0] + 1, undef_p[1] + 2],
                     [undef_p[0] + 2, undef_p[1] - 2], [undef_p[0] + 2, undef_p[1] - 1], [undef_p[0] + 2, undef_p[1]], [undef_p[0] + 2, undef_p[1] + 1], [undef_p[0] + 2, undef_p[1] + 2]])
    # Find the area average intensities of the defined pixels
    for p in area:
        if (p[0] >= 0) and (p[0] < n_rows) and (p[1] >= 0) and (p[1] < n_cols) and (undef_all[p[0], p[1]] == 255):
            intensities.append(polyn_img[p[0], p[1]])
    polyn_img[undef_p[0], undef_p[1]] = int(np.mean(intensities))  # Assign the found intensity

# cv.imwrite(img_path + "all_undefined.jpg", undef_all)
# cv.imwrite(img_path + "borders.jpg", borders)
# cv.imwrite(img_path + "inner_undefined.jpg", undef_inner)

# cv.imshow("All undefined pixels", undef_all)
# cv.imshow("Borders", borders)
# cv.imshow("Inner undefined pixels", undef_inner)

cv.imwrite(img_path + "polyn_winter_manual.jpg", polyn_img)
cv.imshow("Polynomial transform", polyn_img)
cv.waitKey(0)
cv.destroyAllWindows()
