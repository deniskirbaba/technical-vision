# bwareaopen(dim) - removes objects in image A containing less than 'dim' pixels

import cv2 as cv


def bwareopen(img, dim, conn=8):
    if img.ndim > 2:
        return None
    # Find all connected components
    num, labels, stats, centers = cv.connectedComponentsWithStats(img, connectivity=conn)
    # Check size of all connected components
    for i in range(num):
        if stats[i, cv.CC_STAT_AREA] < dim:
            img[labels == i] = 0
    return img

# Reading image
img_path = "C:/denFiles/git/technical-vision/4lab/images/"
img_name = "objects.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Apply bwareaopen
res_img = bwareopen(img, 150, conn=4)

# Show image
cv.imshow("Result", res_img)
cv.waitKey(0)
cv.destroyAllWindows()
