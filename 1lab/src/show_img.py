import cv2 as cv

img_data_folder = "C:/denFiles/git/technical-vision/1lab/images/"

img = cv.imread(img_data_folder + "lake_forest_rain_small.jpg", cv.IMREAD_COLOR)

cv.imshow("image", img)
cv.waitKey(0)
cv.destroyAllWindows()