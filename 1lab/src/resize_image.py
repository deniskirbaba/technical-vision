import cv2 as cv

img = cv.imread('images/lake_forest_rain.jpg', cv.IMREAD_COLOR)
print('Original dimensions: ', img.shape)

scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
new_dim = (width, height)

resized_img = cv.resize(img, new_dim, interpolation=cv.INTER_AREA)
print('Resized dimensions: ', resized_img.shape)

cv.imshow("image", resized_img)
cv.waitKey(0)
cv.destroyAllWindows()