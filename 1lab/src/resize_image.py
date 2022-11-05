import cv2 as cv

img_data_folder = "C:/denFiles/git/technical-vision/2lab/images/"

img = cv.imread(img_data_folder + "winter.jpg", cv.IMREAD_COLOR)
print('Original dimensions: ', img.shape)

scale_percent = 60
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
new_dim = (width, height)

resized_img = cv.resize(img, new_dim, interpolation=cv.INTER_AREA)
print('Resized dimensions: ', resized_img.shape)

cv.imwrite(img_data_folder + "winter_small.jpg", resized_img)
cv.imshow("image", resized_img)
cv.waitKey(0)
cv.destroyAllWindows()