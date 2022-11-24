import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka_quan_noise.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set parameters
# kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
kernel = np.array([[1, 2, 3, 2, 1], [2, 4, 5, 4, 2], [3, 5, 7, 5, 3], [2, 4, 5, 4, 2], [1, 2, 3, 2, 1]])
kernel_shape = kernel.shape

# Convert to float and make image with border
img_copy = img.astype(np.float32) / 255
img_copy = cv.copyMakeBorder(img_copy, int((kernel_shape[0] - 1) / 2), int(kernel_shape[0] / 2), int((kernel_shape[1] - 1) / 2), int(kernel_shape[1] / 2), cv.BORDER_REPLICATE)

# Fill arrays for each kernel item
img_with_areas = np.zeros(img.shape + (np.sum(kernel),), dtype=np.float32)
cur_inx = 0
for i in range(kernel_shape[0]):
    for j in range(kernel_shape[1]):
        # form array with same pixels
        expanded_arr = np.expand_dims(img_copy[i:i + n_rows, j:j + n_cols], axis=2)
        res = expanded_arr
        for k in range(kernel[i, j] - 1):
            res = np.concatenate((res, expanded_arr), axis=2)
        # filling
        img_with_areas[:, :, cur_inx:(cur_inx + kernel[i, j])] = res
        cur_inx += kernel[i, j]

# Sort arrays
img_with_areas.sort()

# Choose layer with concrete rank
img_new = img_with_areas[:, :, np.prod(kernel_shape) // 2]

# Convert back
img_new = np.clip(255 * img_new, 0, 255).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(kernel_shape[0]) + 'x' + str(kernel_shape[1]) + "_weigh_median_fil.jpg", img_new)
cv.imshow("Weighted median filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()
