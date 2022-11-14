import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img = cv.imread(img_path + "oulanka_add_noise.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

kernel_shape = np.array([3, 3])
img_new = np.zeros_like(img, dtype=np.float32)
for i in range(1, n_rows - 1):
    for j in range(1, n_cols - 1):
        img_new[i, j] = np.prod(img[i-1:i+2, j-1:j+2].astype(np.float32))

img_new = np.clip(np.power(img_new, 1 / np.prod(kernel_shape)), 0, 255).astype(np.uint8)

cv.imshow("Geometric averaging filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()