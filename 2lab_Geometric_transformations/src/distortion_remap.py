import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "kitchen.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Form x,y grid
x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))

# Shift and normalize grid
x_mid = n_cols / 2
y_mid = n_rows / 2
x = (x - x_mid) / x_mid
y = (y - y_mid) / y_mid

# Set linear expansion factor
b_0 = 1.005

# Convert to polar and do transformation
r, theta = cv.cartToPolar(x, y)
F3 = 0.3
r_new = (((2/3)**(1/3))*b_0)/((np.sqrt(3*(4*b_0**3*F3**3+27*F3**4*r**2))-9*F3**2*r)**(1/3)) - ((np.sqrt(3*(4*b_0**3*F3**3+27*F3**4*r**2))-9*F3**2*r)**(1/3))/(2**(1/3)*3**(2/3)*F3)

# Undo conversion, normalization and shift
u, v = cv.polarToCart(r_new, theta)
u = u * x_mid + x_mid
v = v * y_mid + y_mid

# Do remapping
dist_img = cv.remap(img, u.astype(np.float32), v.astype(np.float32), cv.INTER_LINEAR)

# Distortion removal
dr_r, dr_theta = cv.cartToPolar((u - x_mid) / x_mid, (v - y_mid) / y_mid)
dr_r_new = dr_r*b_0 + F3*dr_r**3
dr_u, dr_v = cv.polarToCart(dr_r_new, dr_theta)
dr_u = dr_u * x_mid + x_mid
dr_v = dr_v * y_mid + y_mid
dist_rm_img = cv.remap(img, dr_u.astype(np.float32), dr_v.astype(np.float32), cv.INTER_LINEAR)

cv.imwrite(img_path + "kitchen_distortion_remap.jpg", dist_img)
cv.imshow("Distortion removal", dist_rm_img)
cv.imshow("Distortion remap", dist_img)
cv.waitKey(0)
cv.destroyAllWindows()