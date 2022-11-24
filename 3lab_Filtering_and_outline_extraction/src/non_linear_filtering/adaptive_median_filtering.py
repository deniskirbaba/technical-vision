import numpy as np
import cv2 as cv

img_path = "C:/denFiles/git/technical-vision/3lab/images/"
img_name = "oulanka_quan_noise.jpg"
img = cv.imread(img_path + img_name, cv.IMREAD_GRAYSCALE)
n_rows, n_cols = img.shape[:2]

def calculate_intensity(img_padded_float, i, j, s, s_max):
    # Extract window
    window = img_padded_float[i - (s // 2):i + (s // 2) + 1, j - (s // 2):j + (s // 2) + 1]
    # Calculate necessary values
    z_min = np.min(window)
    z_max = np.max(window)
    z_med = np.sort(window.reshape(-1))[window.size // 2]

    # Check condition 1
    if z_min < z_med < z_max:
        h, w = window.shape
        z = window[h // 2, w // 2]
        # Check condition 2
        if z_min < z < z_max:
            return z
        else:
            return z_med
    else:
        # Increase size of the window
        s += 2
        if s <= s_max:
            return calculate_intensity(img_padded_float, i, j, s, s_max)
        else:
            return img_padded_float[i, j]

# Set parameters
s_max = 15
s_initial = 5

# Convert to float and make padding
img_padded_float = img.astype(np.float32) / 255
offset = s_max // 2
img_padded_float = cv.copyMakeBorder(img_padded_float, offset, offset, offset, offset, cv.BORDER_REPLICATE)
filtered_img_padded = np.zeros_like(img_padded_float, dtype=np.float32)

# Go through all pixels
for i in range(offset, n_rows + offset + 1):
    for j in range(offset, n_cols + offset + 1):
        filtered_img_padded[i, j] = calculate_intensity(img_padded_float, i, j, s_initial, s_max)

filtered_img = filtered_img_padded[offset:-offset, offset:-offset]

# Convert back
filtered_img = np.clip(filtered_img * 255, 0, 255).astype(np.uint8)

cv.imwrite(img_path + img_name.rpartition('.')[0] + '_' + str(s_max) + "_ada_med_fil.jpg", filtered_img)
cv.imshow("Adaptive median filtering", filtered_img)
cv.waitKey(0)
cv.destroyAllWindows()
