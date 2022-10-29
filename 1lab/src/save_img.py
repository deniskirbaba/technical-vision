import cv2 as cv

img = cv.imread('images/lake_forest_rain.jpg', cv.IMREAD_COLOR)

scale_percent = 40
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
new_dim = (width, height)

resized_img = cv.resize(img, new_dim, interpolation=cv.INTER_AREA)

cv.imwrite('lake_forest_rain_small.jpg', resized_img)