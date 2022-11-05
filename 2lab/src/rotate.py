import math
import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "winter_small.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

rotation_dot = [50, 50]

angle_grad = 30
angle_rad = angle_grad * math.pi / 180

# T1 = np.float32([[1, 0, -rotation_dot[0]], [0, 1, -rotation_dot[1]], [0, 0, 1]])
# T2 = np.float32([[math.cos(angle_rad), math.sin(angle_rad), 0], [-math.sin(angle_rad), math.cos(angle_rad), 0], [0, 0, 1]])
# T3 = np.float32([[1, 0, rotation_dot[0]], [0, 1, rotation_dot[1]], [0, 0, 1]])
# T = np.matmul(T3, np.matmul(T2, T1))

# using getRotationMatrix func
T = cv.getRotationMatrix2D(rotation_dot, angle_grad, 1)

rotated_img = cv.warpAffine(img, T[0:2, :], (n_cols, n_rows))

cv.imshow("Rotation", np.concatenate((img, rotated_img), axis=1))
cv.waitKey(0)
cv.destroyAllWindows()
