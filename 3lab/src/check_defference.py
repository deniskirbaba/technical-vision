import cv2 as cv

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name_1 = "oulanka_0.5_impulse_noise_3_ada_med_fil.jpg"
img_name_2 = "oulanka_0.5_impulse_noise_21_ada_med_fil.jpg"
img1 = cv.imread(img_path + img_name_1, cv.IMREAD_GRAYSCALE)
img2 = cv.imread(img_path + img_name_2, cv.IMREAD_GRAYSCALE)

mask = img1 == img2
print((~mask).sum())
