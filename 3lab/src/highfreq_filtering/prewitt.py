import cv2 as cv
import numpy as np

# Reading image
img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "lake_grayscale.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set parameters
g_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
g_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Convert to float
img = img.astype(np.float32) / 255

# Apply filtering
img_x = cv.filter2D(img, -1, g_x, borderType=cv.BORDER_REFLECT)
img_y = cv.filter2D(img, -1, g_y, borderType=cv.BORDER_REFLECT)

# Calc magnitude
img_res = cv.magnitude(img_x, img_y)

# Invert colors to best readability
img_res = img_res * (-1) + 1

# Convert back
img_res = np.clip(img_res * 255, 0, 255).astype(np.uint8)

# Save result
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_prewitt.jpg", img_res)

# Show result
cv.imshow("Prewitt filtering", img_res)
cv.waitKey(0)
cv.destroyAllWindows()
