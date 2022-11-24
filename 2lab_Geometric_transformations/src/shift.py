import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)

# Set the parameters
x_shift = -100
y_shift = 60

shifted_img = np.zeros_like(img)

# define the range of indices
if x_shift > 0:
    x_range = range(0, shifted_img.shape[1] - x_shift)
else:
    x_range = range(-x_shift, shifted_img.shape[1])
if y_shift > 0:
    y_range = range(0, shifted_img.shape[0] - y_shift)
else:
    y_range = range(-y_shift, shifted_img.shape[0])

# shifting
for i in y_range:
    for j in x_range:
        shifted_img[i + y_shift, j + x_shift] = img[i, j]

cv.imshow("Shifting image", np.concatenate((img, shifted_img), axis=1))
cv.waitKey(0)
cv.destroyAllWindows()
