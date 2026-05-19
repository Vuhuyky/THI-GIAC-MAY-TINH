import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# ==========================================
# 1. TỰ TẠO ẢNH MẪU ĐỂ CHẠY BÀI TẬP (Tránh lỗi thiếu file ảnh)
# ==========================================
# Tạo một ảnh nền đen 400x400, ở giữa có một hình vuông màu trắng
img = np.zeros((400, 400), dtype=np.uint8)
cv.rectangle(img, (100, 100), (300, 300), 255, -1)

# ==========================================
# 2. ÁP DỤNG CÁC BỘ LỌC ĐẠO HÀM (Cơ bản)
# ==========================================
# Tính Laplacian, Sobel X, Sobel Y bằng kiểu dữ liệu CV_64F để giữ lại số âm
laplacian_raw = cv.Laplacian(img, cv.CV_64F)
sobelx_raw = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
sobely_raw = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)

# Lấy giá trị tuyệt đối và chuyển về uint8 để hiển thị đầy đủ các cạnh
laplacian = np.uint8(np.absolute(laplacian_raw))
sobelx_correct = np.uint8(np.absolute(sobelx_raw))
sobely_correct = np.uint8(np.absolute(sobely_raw))

# ==========================================
# 3. MINH HỌA LỖI MẤT CẠNH (Phần "One Important Matter!")
# ==========================================
# Nếu dùng thẳng kiểu dữ liệu cv.CV_8U (sẽ bị mất cạnh từ Trắng sang Đen)
sobelx_wrong = cv.Sobel(img, cv.CV_8U, 1, 0, ksize=5)

# ==========================================
# 4. TRỰC QUAN HÓA KẾT QUẢ BẰNG MATPLOTLIB
# ==========================================
plt.figure(figsize=(12, 8))

# Hiển thị ảnh gốc
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('1. Anh Goc (Original)'), plt.xticks([]), plt.yticks([])

# Hiển thị Laplacian (Tìm mọi cạnh)
plt.subplot(2, 3, 2)
plt.imshow(laplacian, cmap='gray')
plt.title('2. Bo loc Laplacian'), plt.xticks([]), plt.yticks([])

# Hiển thị Sobel Y (Chỉ tìm cạnh nằm ngang)
plt.subplot(2, 3, 3)
plt.imshow(sobely_correct, cmap='gray')
plt.title('3. Sobel Y (Canh ngang)'), plt.xticks([]), plt.yticks([])

# Hiển thị Sobel X lỗi (Bị mất cạnh bên phải)
plt.subplot(2, 3, 5)
plt.imshow(sobelx_wrong, cmap='gray')
plt.title('4. Sobel X loi (Dung CV_8U)'), plt.xticks([]), plt.yticks([])

# Hiển thị Sobel X đúng (Giữ được cả 2 bên cạnh dọc)
plt.subplot(2, 3, 6)
plt.imshow(sobelx_correct, cmap='gray')
plt.title('5. Sobel X dung (Dung CV_64F)'), plt.xticks([]), plt.yticks([])

print("Đang hiển thị biểu đồ so sánh các bộ lọc...")
plt.tight_layout()
plt.show()