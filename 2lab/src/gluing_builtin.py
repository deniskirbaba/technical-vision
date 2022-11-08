import cv2 as cv

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img_top = cv.imread(img_path + "rowan_top.jpg", cv.IMREAD_COLOR)
img_bot = cv.imread(img_path + "rowan_bot.jpg", cv.IMREAD_COLOR)

stitcher = cv.Stitcher.create(cv.STITCHER_SCANS)
status, stitched_img = stitcher.stitch([img_top, img_bot])

cv.imshow("Stitched image", stitched_img)
cv.waitKey(0)
cv.destroyAllWindows()