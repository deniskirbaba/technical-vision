import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Form the transformation matrix with reflect and shift
T = np.float32([[1, 0, 0], [0, -1, n_rows - 1]])
# Apply transform
# reflected_img = cv.warpAffine(img, T, (n_cols, n_rows))

# Using flip
reflected_img = cv.flip(img, 0)

cv.imwrite(img_path + "reflected_winter.jpg", reflected_img)
cv.imshow("Reflected image", np.concatenate((img, reflected_img), axis=1))
cv.waitKey(0)
cv.destroyAllWindows()
