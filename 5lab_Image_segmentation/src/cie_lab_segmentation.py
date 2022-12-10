import cv2 as cv
import numpy as np

# Reading image
img_path = "C:/Users/R1505-W-2-Stud/Downloads/technical-vision-main/technical-vision-main/5lab_Image_segmentation/images/"
img_name = "colorful_apples.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_COLOR)

# Convert to CIE Lab color space and then split into components
img_lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
img_lab_l, img_lab_a, img_lab_b = cv.split(img_lab)

# Set number of color segments
col_seg_num = 3

# Define areas
sample_areas = []
def mouse_handler(event, x, y, flags, param):
    if event != cv.EVENT_LBUTTONDBLCLK:
        return
    sample_areas.append((x, y))

cv.imshow("Image", img)
cv.setMouseCallback("Image", mouse_handler)
while len(sample_areas) < col_seg_num:  # Interactive loop
    cv.waitKey(20)
cv.setMouseCallback("Image", lambda *args : None)
cv.destroyAllWindows()

# Compute means in the circle areas of selected pixels
color_marks = []
color_marks_bgr = []
for pix in sample_areas:
    mask = np.zeros_like(img_lab_l)
    cv.circle(mask, pix, 10, 255, -1)  # Draw circle
    a = img_lab_a.mean(where = mask > 0)
    b = img_lab_b.mean(where = mask > 0)
    color_marks.append((a, b))
    color_marks_bgr.append(img[mask > 0, :].mean(axis=0))

# Calculate the distance between all pixels in initial image and previously selected pixels
distance = []
for color in color_marks:
    distance.append(np.sqrt(np.power(img_lab_a - color[0], 2) + np.power(img_lab_b - color[1], 2)))

# Estimate minimum of all pixels
distance_min = np.minimum.reduce(distance, axis=0)

# Find labels of all pixels
labels = np.zeros_like(img[:, :, 0], dtype=np.uint8)
for i in range(len(color_marks)):
    mask = distance_min == distance[i]
    labels[mask] = i

# Segment the source image to a set of segmented images
segmented_frames = []
for i in range(len(color_marks)):
    img_buf = np.zeros_like(img)
    mask = labels == i
    img_buf[mask] = img[mask]
    segmented_frames.append(img_buf)

# Plot distribution of image pixel colors in (a, b) coordinate system
img_plot = np.full((256, 256, 3), 255, dtype=np.uint8)
for i in range(len(color_marks)):
    img_buf = np.zeros_like(img)
    mask = labels == i
    img_plot[img_lab_a[mask], img_lab_b[mask], :] = color_marks_bgr[i]

# Show distribution plot
cv.imshow("Distribution plot", img_plot)
cv.waitKey(0)
cv.destroyAllWindows()

# Show segmented frames
for i in range(col_seg_num):
    cv.imshow(str(i) + " segment", segmented_frames[i])
cv.waitKey(0)
cv.destroyAllWindows()

print("End of the program...")
