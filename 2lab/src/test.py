import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "guitar_left.jpg", cv.IMREAD_GRAYSCALE)

cv.imwrite(img_path + "guitar_left_gray.jpg", img)