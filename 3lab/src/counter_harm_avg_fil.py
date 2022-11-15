import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka_1_impulse_noise.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set parameters
Q = -0.5
kernel_shape = np.array([3, 3])
kernel = np.ones(kernel_shape, dtype=np.float32)
div_state = np.seterr(divide='ignore')

# Convert to float and make image with border
img_copy = img.astype(np.float32) / 255
img_copy = cv.copyMakeBorder(img_copy, int((kernel_shape[0] - 1) / 2), int(kernel_shape[0] / 2), int((kernel_shape[1] - 1) / 2), int(kernel_shape[1] / 2), cv.BORDER_REFLECT)

# Create four auxiliary arrays
buf1 = np.power(img_copy, Q + 1)
buf2 = np.power(img_copy, Q)
numerator = np.zeros_like(img)
denumerator = np.zeros_like(img)

img_new = np.zeros_like(img)
# Filtering
for i in range(kernel_shape[0]):
    for j in range(kernel_shape[1]):
        numerator = numerator + buf1[i:i + n_rows, j:j + n_cols]
        denumerator = denumerator + buf2[i:i + n_rows, j:j + n_cols]
img_new = np.divide(numerator, denumerator)

# Convert back
img_new = np.clip(img_new * 255, 0, 255).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(kernel_shape[0]) + 'x' + str(kernel_shape[1]) + '_' + str(Q) + "_counter_harm_avg_fil.jpg", img_new)
cv.imshow("Counter-harmonic averaging filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()

# ---------------------------------------------------
# Other example of non-vectorized calculation. More complex and time-consuming calculations.
#
# img_new = np.zeros_like(img, dtype=np.float64)
# buf1 = np.power(img.astype(np.float64), Q+1)
# buf2 = np.power(img.astype(np.float64), Q)
#
# for i in range(kernel_shape[0] // 2, n_rows - (kernel_shape[1] // 2)):
#     for j in range(kernel_shape[0] // 2, n_cols - (kernel_shape[1] // 2)):
#         numerator = np.sum(buf1[(i-(kernel_shape[0]//2)):(i+(kernel_shape[0]//2)+1), (j-(kernel_shape[1]//2)):(j+(kernel_shape[0]//2)+1)])
#         denumerator = np.sum(buf2[(i-(kernel_shape[0]//2)):(i+(kernel_shape[0]//2)+1), (j-(kernel_shape[1]//2)):(j+(kernel_shape[0]//2)+1)])
#         img_new[i, j] = numerator / denumerator
#
# ---------------------------------------------------
