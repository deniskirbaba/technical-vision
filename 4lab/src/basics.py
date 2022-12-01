import cv2 as cv
import numpy as np

# Create struct element
el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# Reading image
img_path = "C:/denFiles/git/technical-vision/4lab/images/"
img_name = "objects.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Apply morph
# res_img = cv.morphologyEx(img, cv.MORPH_HITMISS, el, iterations=1)

# Create connected components list
# retval, labels = cv.connectedComponents(img)

# Create connected components list with stats
retval, labels, stats, centroids = cv.connectedComponentsWithStats(img)

# Show image
# cv.imshow("Result", res_img)
# cv.waitKey(0)
# cv.destroyAllWindows()
