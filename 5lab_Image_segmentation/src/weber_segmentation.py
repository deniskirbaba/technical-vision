import cv2 as cv
import numpy as np


def weber(i):
    if i <= 88:
        return 20 - 12 * i / 88
    elif i <= 138:
        return 0.002 * np.sqrt(i - 88)
    else:
        return 7 * (i - 138) / 117 + 13


# Reading image
img_path = "C:/denFiles/git/technical-vision/5lab_Image_segmentation/images/"
img_name = "mandarin.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Initial conditions
n = 1  # First class number
i_n = 0  # Grayscale level
i_w = weber(i_n)
i_max = np.max(img)

new_img = np.zeros_like(img)

while i_max > i_w:
    new_img[(img > i_n) & (img <= i_n + i_w)] = i_n
    i_n += (i_w + 1)
    n += 1
    i_w = weber(i_n)

# Show image
# cv.imwrite(img_path + img_name.rpartition('.')[0] + "_black_bg.jpg", img)
cv.imshow("Result", new_img)
cv.waitKey(0)
cv.destroyAllWindows()

print("Done")
