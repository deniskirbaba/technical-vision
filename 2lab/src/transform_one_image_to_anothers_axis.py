import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img_1 = cv.imread(img_path + "guitar_top.jpg", cv.IMREAD_GRAYSCALE)
img_2 = cv.imread(img_path + "guitar_bot.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img_1.shape[:2]

# Dots
d1 = [[516, 1259], [372, 778], [566, 571]]
d2 = [[526, 779], [374, 271], [580, 84]]

# Find parameters of the transform
# To translate the top image into the bot coordinate system
buf = np.linalg.inv([[1, d1[0][0], d1[0][1]], [1, d1[1][0], d1[1][1]],
                     [1, d1[2][0], d1[2][1]]])
x_param = np.matmul(buf, np.reshape([d2[0][0], d2[1][0], d2[2][0]], (3, 1)))
y_param = np.matmul(buf, np.reshape([d2[0][1], d2[1][1], d2[2][1]], (3, 1)))

# Create grid corresponding to our pixels
x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

# Calculate all new X and Y coordinates
x_new = np.round(x_param[0, 0] + x_param[1, 0] * x + x_param[2, 0] * y).astype(np.float32)
y_new = np.round(y_param[1, 0] * x + y_param[2, 0] * y).astype(np.float32) # Do notuse y_param[0, 0] because we don't need oY shift

# Calculate mask for valid indices
mask = np.logical_and(np.logical_and(x_new >= 0, x_new < n_cols), np.logical_and(y_new >= 0, y_new < n_rows))

# Apply reindexing
new_img = np.zeros_like(img_1)
new_img[y_new[mask].astype(int), x_new[mask].astype(int)] = img_1[y[mask], x[mask]]

# Find undefined pixels and borders of new image
undef_all = np.zeros_like(img_1)
borders = np.zeros_like(img_1)
undef_inner = np.zeros_like(img_1)
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
    # Construct 3x3 punctured area
    area = np.array([[undef_p[0] - 1, undef_p[1] - 1], [undef_p[0] - 1, undef_p[1]], [undef_p[0] - 1, undef_p[1] + 1],
                     [undef_p[0], undef_p[1] - 1], [undef_p[0], undef_p[1] + 1],
                     [undef_p[0] + 1, undef_p[1] - 1], [undef_p[0] + 1, undef_p[1]], [undef_p[0] + 1, undef_p[1] + 1]])
    # Find the area average intensities of the defined pixels
    for p in area:
        if (p[0] >= 0) and (p[0] < n_rows) and (p[1] >= 0) and (p[1] < n_cols) and (undef_all[p[0], p[1]] == 255):
            intensities.append(new_img[p[0], p[1]])
    new_img[undef_p[0], undef_p[1]] = int(np.mean(intensities))  # Assign the found intensity

cv.imwrite(img_path + "new_guitar_top.jpg", new_img)
cv.imshow("New left guitar image", new_img)
cv.waitKey(0)
cv.destroyAllWindows()
