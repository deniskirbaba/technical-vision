import cv2 as cv

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img_top = cv.imread(img_path + "guitar_l.jpg", cv.IMREAD_GRAYSCALE)
img_bot = cv.imread(img_path + "guitar_r.jpg", cv.IMREAD_GRAYSCALE)

stitcher = cv.Stitcher.create(cv.STITCHER_PANORAMA)
status, stitched_img = stitcher.stitch([img_top, img_bot])

cv.imshow("Stitched image", stitched_img)
cv.waitKey(0)
cv.destroyAllWindows()