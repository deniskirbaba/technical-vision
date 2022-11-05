import cv2 as cv
import numpy as np
import math

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

amp = 20
freq_factor = 2
phase = 0

x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
x = x + amp * np.sin((freq_factor * math.pi * y + phase) / 180)

sin_img = cv.remap(img, x.astype(np.float32), y.astype(np.float32), cv.INTER_NEAREST)

cv.imshow("Sinusoid transform", sin_img)
cv.waitKey(0)
cv.destroyAllWindows()
