import cv2 as cv

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img = cv.imread(img_path + "oulanka_add_noise.jpg", cv.IMREAD_GRAYSCALE)

img_new = cv.blur(img, (3, 3))

cv.imshow("Arithmetical averaging filtering", img_new)
cv.waitKey(0)
cv.destroyAllWindows()