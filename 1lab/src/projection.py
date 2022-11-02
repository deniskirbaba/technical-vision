import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
img = cv.imread(src_path + "text.png", cv.IMREAD_GRAYSCALE)

# calculate projection for oY
y_proj = np.sum(img, axis=1)

plt.plot(y_proj, np.arange(y_proj.shape[0] - 1, -1, -1))
plt.show()

cv.imshow("text", img)
cv.waitKey(0)
cv.destroyAllWindows()
