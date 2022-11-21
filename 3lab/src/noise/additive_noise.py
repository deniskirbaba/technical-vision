import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)

# Convert to float and normalize
img_float = img.astype(np.float32) / 255

# Generate random numbers using exp distribution
rng = np.random.default_rng()
exp = rng.exponential(5, img.shape)
exp = exp / np.max(exp)  # Normalize

# Adding noise and convert to uint8
img_new = np.clip((img_float + exp) * 255, 0, 255).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + "_add_noise.jpg", img_new)
cv.imshow("Additive noise", img_new)
cv.waitKey(0)
cv.destroyAllWindows()