import cv2 as cv
import numpy as np

# Reading image
img_path = "C:/Files/git/technical-vision/5lab_Image_segmentation/images/"
img_name = "colorful_apples.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_COLOR)

# Convert to CIE Lab color space and then split into components
img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
img_lab_l, img_lab_a, img_lab_b = cv.split(img_lab)

# Set number of color segments
col_seg_num = 3

# Merge 'a' and 'b' components and then reshape it
ab = cv.merge([img_lab_a, img_lab_b])
ab = ab.reshape(-1, 2).astype(np.float32)

# Apply k-means clustering
# Stop criteria is defined as not more than 10 iterations of difference between steps less than 1
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#  Starting points are selected randomly (due to cv2.KMEANS_RANDOM_CENTERS is used) and selection is done 10 times
ret, labels, centers = cv.kmeans(ab, col_seg_num, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
# Reshape labels back to an original image
labels = labels.reshape(img_lab_l.shape)

# Segment image to a set of images
segmented_frames = []
for i in range(col_seg_num):
    mask = labels == i
    img_buf = np.zeros_like(img)
    img_buf[mask] = img[mask, :]
    segmented_frames.append(img_buf)

# Show segmented frames
for i in range(col_seg_num):
    cv.imshow(str(i) + " segment", segmented_frames[i])
cv.waitKey(0)
cv.destroyAllWindows()

print("End of the program...")
