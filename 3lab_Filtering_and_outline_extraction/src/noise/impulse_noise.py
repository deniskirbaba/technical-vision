import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

# Noise parameter
p = 0.01
# Salt vs pepper ratio
s_p_ratio = 0.7
# Generate random numbers
rng = np.random.default_rng()
vals = rng.random(img.shape)

img_new = np.copy(img)
img_new[vals < p * s_p_ratio] = 255  # Add salt
img_new[np.logical_and(vals >= p * s_p_ratio, vals < p)] = 0  # Add pepper

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(s_p_ratio) + "_impulse_noise.jpg", img_new)
cv.imshow("Impulse noise", img_new)
cv.waitKey(0)
cv.destroyAllWindows()