import cv2 as cv
import numpy as np

# 1. PHÉP CỘNG ẢNH
img1 = cv.imread('ml.png')
img2 = cv.imread('opencv-logo.png')

assert img1 is not None, "Không đọc được img1"
assert img2 is not None, "Không đọc được img2"

# Resize để cùng kích thước 
img2_resize = cv.resize(img2, (img1.shape[1], img1.shape[0]))

# Cộng bằng OpenCV 
add_cv = cv.add(img1, img2_resize)

# Cộng bằng numpy 
add_np = img1 + img2_resize

cv.imshow("Add OpenCV", add_cv)
cv.imshow("Add Numpy", add_np)

# 2. BLENDING
blend = cv.addWeighted(img1, 0.7, img2_resize, 0.3, 0)

cv.imshow("Blended Image", blend)

# 3. BITWISE
# Đọc lại ảnh gốc 
img1 = cv.imread('messi5.jpg')
img2 = cv.imread('opencv-logo.png')

assert img1 is not None, "Không đọc được img1"
assert img2 is not None, "Không đọc được img2"

# resize logo nhỏ lại
img2 = cv.resize(img2, (300, 300))

# Lấy kích thước logo
rows, cols, channels = img2.shape

# Tạo ROI trên ảnh nền
roi = img1[0:rows, 0:cols]

# TẠO MASK
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)

mask_inv = cv.bitwise_not(mask)

# XỬ LÝ ẢNH
# Xóa vùng logo trên nền
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)

# Lấy riêng logo
img2_fg = cv.bitwise_and(img2, img2, mask=mask)

# GHÉP ẢNH
dst = cv.add(img1_bg, img2_fg)

# Ghi lại vào ảnh chính
img1[0:rows, 0:cols] = dst

# HIỂN THỊ

cv.imshow("Final Result", img1)

cv.waitKey(0)
cv.destroyAllWindows()