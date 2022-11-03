import cv2 as cv
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
# reading a barcode
img = cv.imread(src_path + "barcode.jpg", cv.IMREAD_GRAYSCALE)

# make a profile array, using slice of the initial img array
profile = img[int(img.shape[0] / 2), :]

plt.plot(profile)
plt.show()

cv.imshow("barcode", img)
cv.waitKey(0)
cv.destroyAllWindows()
