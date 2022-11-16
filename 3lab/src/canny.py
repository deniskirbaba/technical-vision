import cv2 as cv
import numpy as np

# Set parameters
t1 = 200
t2 = 250

# Reading image
img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Apply Canny algorithm
img_res = cv.Canny(img, t1, t2)

# Invert colors for better readability
img_res = (img_res.astype(np.int16) * (-1) + 255).astype(np.uint8)

# Save result
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_canny.jpg", img_res)

# Show result
cv.imshow("Canny algorithm", img_res)
cv.waitKey(0)
cv.destroyAllWindows()
