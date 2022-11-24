import cv2 as cv
import numpy as np

img_folder = "C:/denFiles/git/technical-vision/1lab/images/"

img = cv.imread(img_folder + "snowforest_small.jpg", cv.IMREAD_COLOR)
img2 = cv.imread(img_folder + "snowforest_small_gray.jpg", cv.IMREAD_COLOR)

imgs = np.concatenate((img, img2), axis=1)

cv.imshow("Images", imgs)
cv.waitKey(0)
cv.destroyAllWindows()