import numpy as np

img = np.arange(0, 9).reshape((3, 3))
img_with_areas = np.zeros(img.shape + (3, ))
qqq = np.expand_dims(img, axis=2)
res = qqq
for k in range(5):
    res = np.concatenate((res, qqq), axis=2)

kernel = np.array([[1, 2, 3, 2, 1], [2, 4, 5, 4, 2], [3, 5, 7, 5, 3], [2, 4, 5, 4, 2], [1, 2, 3, 2, 1]])
print(np.sum(kernel))

import cv2 as cv

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "rowan.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
cv.imwrite(img_path + "rowan_grayscale.jpg", img)
