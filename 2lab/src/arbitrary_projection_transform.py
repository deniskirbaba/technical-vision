import cv2 as cv
import numpy as np

img_path = "C:/denFiles/git/technical-vision/2lab/images/"
img = cv.imread(img_path + "guitar_left_gray.jpg", cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

# Set the transformation matrix
# T = np.float32([[1.65, 0.3, 0], [0.2, 1.7, 0], [0.001, 0.003, 1]])

# or using builtin func
pts_src = np.float32([[775, 881], [871, 252], [888, 1152], [958, 1058]])
pts_dst = np.float32([[58, 753], [104, 54], [159, 1000], [239, 892]])
T = cv.getPerspectiveTransform(pts_src, pts_dst)

# Apply the transformation
new_img = cv.warpPerspective(img, T, (n_cols, n_rows))

# cv.imwrite(img_path + "arbitrary_proj_winter.jpg", new_img)
cv.imshow("Arbitrary projective transform", new_img)
cv.waitKey(0)
cv.destroyAllWindows()