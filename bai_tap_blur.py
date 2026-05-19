import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# ==========================================
# 1. TỰ TẠO ẢNH MẪU CÓ NHIỄU (Tránh lỗi thiếu ảnh)
# ==========================================
# Tạo ảnh nền xám có hình tròn và chữ
img = np.ones((300, 400), dtype=np.uint8) * 180
cv.circle(img, (150, 150), 70, 50, -1)
cv.putText(img, 'OpenCV', (70, 270), cv.FONT_HERSHEY_SIMPLEX, 2, 255, 4, cv.LINE_AA)

# Thêm nhiễu "Muối tiêu" (Salt & Pepper Noise) để thử nghiệm bộ lọc
noisy_img = img.copy()
num_salt = 1500
# Điểm muối (màu trắng)
coords_salt = [np.random.randint(0, i - 1, num_salt) for i in noisy_img.shape]
noisy_img[tuple(coords_salt)] = 255
# Điểm tiêu (màu đen)
coords_pepper = [np.random.randint(0, i - 1, num_salt) for i in noisy_img.shape]
noisy_img[tuple(coords_pepper)] = 0

# ==========================================
# 2. ÁP DỤNG 4 PHƯƠNG PHÁP LÀM MỊN ẢNH
# ==========================================
# Cấu hình kích thước bộ lọc (Kernel size) là 9x9 để thấy rõ hiệu ứng
k_size = 9

# Cách 1: Averaging (Bộ lọc trung bình)
blur_avg = cv.blur(noisy_img, (k_size, k_size))

# Cách 2: Gaussian Blurring
blur_gaussian = cv.GaussianBlur(noisy_img, (k_size, k_size), 0)

# Cách 3: Median Blurring (Khắc tinh của nhiễu muối tiêu)
blur_median = cv.medianBlur(noisy_img, k_size)

# Cách 4: Bilateral Filtering (Giữ nét đường biên)
# Lưu ý: Hàm này yêu cầu ảnh màu hoặc ảnh uint8 chuẩn, ta truyền vào các tham số d, sigmaColor, sigmaSpace
blur_bilateral = cv.bilateralFilter(noisy_img, d=9, sigmaColor=75, sigmaSpace=75)

# ==========================================
# 3. TRỰC QUAN HÓA VÀ SO SÁNH
# ==========================================
plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(noisy_img, cmap='gray')
plt.title('1. Anh bi nhiễu gốc'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 2)
plt.imshow(blur_avg, cmap='gray')
plt.title('2. Averaging (Mờ biên)'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 3)
plt.imshow(blur_gaussian, cmap='gray')
plt.title('3. Gaussian Blur'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 5)
plt.imshow(blur_median, cmap='gray')
plt.title('4. Median (Sạch nhiễu 100%)'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 6)
plt.imshow(blur_bilateral, cmap='gray')
plt.title('5. Bilateral (Giữ nét cạnh)'), plt.xticks([]), plt.yticks([])

print("Đang hiển thị kết quả so sánh các bộ lọc làm mịn...")
plt.tight_layout()
plt.show()