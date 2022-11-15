import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka_1_impulse_noise.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

kernel_shape = np.array([3, 3])
img_new = np.zeros_like(img, dtype=np.float64)
div_state = np.seterr(divide='ignore')
buf = np.divide(1.0, img.astype(np.float64))
for i in range(kernel_shape[0] // 2, n_rows - (kernel_shape[1] // 2)):
    for j in range(kernel_shape[0] // 2, n_cols - (kernel_shape[1] // 2)):
        img_new[i, j] = 1 / np.sum(buf[(i-(kernel_shape[0]//2)):(i+(kernel_shape[0]//2)+1), (j-(kernel_shape[1]//2)):(j+(kernel_shape[0]//2)+1)])
img_new *= np.prod(kernel_shape)

img_new = np.clip(img_new, 0, 255).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(kernel_shape[0]) + 'x' + str(kernel_shape[1]) + "_harm_avg_fil.jpg", img_new)
cv.imshow("Harmonic averaging filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()
