import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

src_path = "C:/denFiles/git/technical-vision/1lab/images/"
# reading an image
img = cv.imread(src_path + "text.png", cv.IMREAD_GRAYSCALE)

# calculate projection for oX
x_proj = np.sum(img, axis=0)
# calculate projection for oY
y_proj = np.sum(img, axis=1)

# calculate the oX, oY boundaries
margin = 255  # the value in the array must change by at least "margin"
# so that we can assume that the void has been replaced by text or vice versa
x_boundaries = []
for i in range(x_proj.size - 1):
    if np.abs(x_proj[i].astype(np.int32) - x_proj[i + 1]) > margin:
        x_boundaries.append(i)
y_boundaries = []
for i in range(y_proj.size - 1):
    if np.abs(y_proj[i].astype(np.int32) - y_proj[i + 1]) > margin:
        y_boundaries.append(i)

# plot the projections graphs
plt.rcParams["figure.figsize"] = (16, 8)
figure, axis = plt.subplots(1, 2)
axis[0].plot(np.arange(x_proj.shape[0]), x_proj, 'r', label="oX projection")
axis[0].plot([x_boundaries[0], x_boundaries[0]], [min(x_proj), max(x_proj)], 'g', linestyle="dashed")
axis[0].plot([x_boundaries[-1], x_boundaries[-1]], [min(x_proj), max(x_proj)], 'g', linestyle="dashed")

axis[1].plot(y_proj, np.arange(y_proj.shape[0] - 1, -1, -1), 'b', label="oY projection")
axis[1].plot([min(y_proj), max(y_proj)], [y_boundaries[0], y_boundaries[0]], 'g', linestyle="dashed")
axis[1].plot([min(y_proj), max(y_proj)], [y_boundaries[-1], y_boundaries[-1]], 'g', linestyle="dashed")
figure.legend(loc="upper right")
plt.show()

start_point = (x_boundaries[0], img.shape[0] - y_boundaries[-1])  # top left corner
end_point = (x_boundaries[-1], img.shape[0] - y_boundaries[0])  # bottom right corner
color = (0, 0, 255)  # color in BGR
thickness = 1  # line thickness

# draw a calculated borders of the object on the image
img = cv.rectangle(img, start_point, end_point, color, thickness)

cv.imshow("text", img)
cv.imwrite(src_path + "text_rect.jpg", img)
cv.waitKey(0)
cv.destroyAllWindows()
