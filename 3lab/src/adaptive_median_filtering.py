# import cv2 as cv
# import numpy as np
#
# img_path = "C:/denFiles/git/technical-vision/3lab/images/"
# img_name = "balloons_noisy.png"
# img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
# n_rows, n_cols = img.shape[:2]
#
# # Set parameters
# s_max = 10  # Maximum allowable kernel size
# kernel_shape = np.array([3, 3])
# rank = 4
# kernel = np.ones(kernel_shape, dtype=np.float32)
#
# # Stopping point array
# mask = np.full_like(img, False, dtype=bool)
#
# while (~mask).all():
#     for i in range(kernel_shape[0]):
#         for j in range(kernel_shape[1]):
#
