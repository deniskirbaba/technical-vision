import cv2 as cv

img = cv.imread('images/lake_forest_rain.jpg', cv.IMREAD_COLOR)

cv.imshow("image", img)
cv.waitKey(0)
cv.destroyAllWindows()