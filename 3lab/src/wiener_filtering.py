import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka_0.5_impulse_noise.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set parameters
kernel_shape = np.array([3, 3])
kernel = np.ones(kernel_shape, dtype=np.float32)

# Convert to float and make image with border
img_copy = img.astype(np.float32) / 255
img_copy = cv.copyMakeBorder(img_copy, int((kernel_shape[0] - 1) / 2), int(kernel_shape[0] / 2), int((kernel_shape[1] - 1) / 2), int(kernel_shape[1] / 2), cv.BORDER_REPLICATE)

k_squared = np.power(kernel, 2)

# Calculate temporary matrices for img ** 2
img_squared = np.power(img_copy, 2)
m = np.zeros(img.shape[:2], np.float32)
q = np.zeros(img.shape[:2], np.float32)
# Calculate variance values
for i in range(kernel_shape[0]):
    for j in range(kernel_shape[1]):
        m = m + kernel[i, j] * img_copy[i:i + n_rows, j:j + n_cols]
        q = q + k_squared[i, j] * img_squared[i:i + n_rows, j:j + n_cols]
m = m / np.sum(kernel)
q = q / np.sum(kernel)
q = q - m * m

# Calculate noise as an average variance
v = np.sum(q) / img.size

# Do filtering
img_new = img_copy[(kernel_shape[0] - 1) // 2:(kernel_shape[0] - 1) // 2 + n_rows, (kernel_shape[1] - 1) // 2:(kernel_shape[1] - 1) // 2 + n_cols]
img_new = np.where(q < v, m, (img_new - m) * (1 - v / q) + m)

# Convert back
img_new = np.clip(255 * img_new, 0, 255).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(kernel_shape[0]) + 'x' + str(kernel_shape[1]) + "_wiener_fil.jpg", img_new)
cv.imshow("Wiener filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()
