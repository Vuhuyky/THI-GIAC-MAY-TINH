import cv2 as cv
import numpy as np

# Đọc ảnh thay vì dùng webcam
img = cv.imread('color_test.jpg')

assert img is not None, "Không tìm thấy file color_test.jpg, hãy kiểm tra lại đường dẫn!"
# img = cv.resize(img, (600, 400)) 

# Chuyển đổi từ không gian màu BGR sang HSV
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

# Màu Xanh lam (Blue) 
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])
mask_blue = cv.inRange(hsv, lower_blue, upper_blue)

# Màu Xanh lục (Green) 
lower_green = np.array([35, 100, 50])
upper_green = np.array([85, 255, 255])
mask_green = cv.inRange(hsv, lower_green, upper_green)

# Màu Đỏ (Red) 
# Trong HSV, màu đỏ nằm ở 2 đầu (0-10 và 170-180) 
# nên phải tạo 2 mask và gộp lại.
lower_red1 = np.array([0, 150, 50])
upper_red1 = np.array([10, 255, 255])
mask_red1 = cv.inRange(hsv, lower_red1, upper_red1)

lower_red2 = np.array([170, 150, 50])
upper_red2 = np.array([180, 255, 255])
mask_red2 = cv.inRange(hsv, lower_red2, upper_red2)

mask_red = cv.bitwise_or(mask_red1, mask_red2)

mask_combined = cv.bitwise_or(mask_blue, mask_green)
mask_combined = cv.bitwise_or(mask_combined, mask_red)

res = cv.bitwise_and(img, img, mask=mask_combined)

cv.imshow('1. Original Image', img)
cv.imshow('2. Combined Mask', mask_combined)
cv.imshow('3. Final Result', res)

cv.waitKey(0)
cv.destroyAllWindows()