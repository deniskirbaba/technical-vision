import cv2 as cv

# Reading image
img_path = "C:/denFiles/git/technical-vision/4lab/images/"
img_name = "circles_defects_opening.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Create struct element
el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))

# Apply morph
res_img = cv.morphologyEx(img, cv.MORPH_CLOSE, el, borderType=cv.BORDER_CONSTANT, borderValue=255)

# Save result
cv.imwrite(img_path + img_name.rpartition('.')[0] + "_closing.jpg", res_img)

# Show image
cv.imshow("Result", res_img)
cv.waitKey(0)
cv.destroyAllWindows()
