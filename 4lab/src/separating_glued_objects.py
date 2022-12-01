import cv2 as cv
import numpy as np

# Reading image
img_path = "C:/denFiles/git/technical-vision/4lab/images/"
img_name = "oranges.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

ret, thresholding_img = cv.threshold(img, 70, 255, cv.THRESH_BINARY)
cv.imshow("Thresholding image", thresholding_img)
cv.waitKey(0)
cv.destroyAllWindows()

el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

# Erosion
working_img = cv.morphologyEx(thresholding_img, cv.MORPH_ERODE, el, iterations=3, borderType=cv.BORDER_CONSTANT, borderValue=(0))
cv.imshow("Erosed image", working_img)
cv.waitKey(0)
cv.destroyAllWindows()

# Dilation
img_borders = np.zeros_like(img)
while cv.countNonZero(working_img) < working_img.size:
    dilated_img = cv.morphologyEx(working_img, cv.MORPH_DILATE, el, borderType=cv.BORDER_CONSTANT, borderValue=(0))
    closed_img = cv.morphologyEx(dilated_img, cv.MORPH_CLOSE, el, borderType=cv.BORDER_CONSTANT, borderValue=(0))
    substracted_img = closed_img - dilated_img
    img_borders = cv.bitwise_or(substracted_img, img_borders)
    working_img = dilated_img

# Closing for borders
img_borders = cv.morphologyEx(img_borders, cv.MORPH_CLOSE, el, iterations=10, borderType=cv.BORDER_CONSTANT, borderValue=(255))

# Remove borders from image
img_res = cv.bitwise_and(~img_borders, thresholding_img)

# Show image
cv.imshow("Borders", img_borders)
cv.imshow("Result", img_res)
cv.waitKey(0)
cv.destroyAllWindows()
