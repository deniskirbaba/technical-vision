import cv2 as cv
import numpy as np

# Reading image
img_path = "C:/denFiles/git/technical-vision/4lab/images/"
img_name = "circles_defects.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Create struct element
el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# Apply morph
res_img = cv.morphologyEx(img, cv.MORPH_ERODE, el, borderType=cv.BORDER_CONSTANT, borderValue=255)

# Create connected components list
# retval, labels = cv.connectedComponents(img)

# Create connected components list with stats
# retval, labels, stats, centroids = cv.connectedComponentsWithStats(img)

# Show image
cv.imshow("Result", res_img)
cv.waitKey(0)
cv.destroyAllWindows()