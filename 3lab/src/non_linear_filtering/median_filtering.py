import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka_quan_noise.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set parameters
kernel_shape = np.array([5, 5])

# Convert to float and make image with border
img_copy = img.astype(np.float32) / 255
img_copy = cv.copyMakeBorder(img_copy, int((kernel_shape[0] - 1) / 2), int(kernel_shape[0] / 2), int((kernel_shape[1] - 1) / 2), int(kernel_shape[1] / 2), cv.BORDER_REPLICATE)

# Fill arrays
img_with_areas = np.zeros(img.shape + (np.prod(kernel_shape),), dtype=np.float32)
for i in range(kernel_shape[0]):
    for j in range(kernel_shape[1]):
        img_with_areas[:, :, i * kernel_shape[1] + j] = img_copy[i:i + n_rows, j:j + n_cols]

# Sort arrays
img_with_areas.sort()

# Choose median pixel
img_new = img_with_areas[:, :, np.prod(kernel_shape) // 2]

# Convert back
img_new = np.clip(255 * img_new, 0, 255).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(kernel_shape[0]) + 'x' + str(kernel_shape[1]) + "_median_fil.jpg", img_new)
cv.imshow("Median filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()
