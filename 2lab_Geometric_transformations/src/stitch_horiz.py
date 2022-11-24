import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img_left = cv.imread(img_path + "guitar_l.jpg", cv.IMREAD_GRAYSCALE)
img_right = cv.imread(img_path + "guitar_r.jpg", cv.IMREAD_GRAYSCALE)

# Match template
template_size = 30
template = img_left[:, -template_size:]
res = cv.matchTemplate(img_right, template, cv.TM_CCOEFF)

min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

full_img = np.zeros((img_left.shape[0], img_left.shape[1] + img_right.shape[1] - max_loc[0] - template_size), dtype=np.uint8)
full_img[:, :img_left.shape[1]] = img_left
full_img[:, img_left.shape[1]:] = img_right[:, max_loc[0] + template_size:]

cv.imshow("Glued image", full_img)
cv.waitKey(0)
cv.destroyAllWindows()
