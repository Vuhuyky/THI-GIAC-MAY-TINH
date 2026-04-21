import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# ==========================================================
# BƯỚC 1: CHUẨN BỊ ẢNH
# ==========================================================
# Thay tên file ảnh của bạn vào đây
path = 'messi5.jpg' 
img = cv.imread(path, 0) # Đọc ảnh ở chế độ xám (Grayscale)

if img is None:
    print("Lỗi: Không tìm thấy ảnh. Vui lòng kiểm tra lại đường dẫn!")
    exit()

# ==========================================================
# BƯỚC 2: TÍNH HISTOGRAM THỦ CÔNG (KHÔNG DÙNG THƯ VIỆN)
# ==========================================================
h, w = img.shape
hist_manual = [0] * 256

for i in range(h):
    for j in range(w):
        pixel_value = img[i, j]
        hist_manual[pixel_value] += 1

# ==========================================================
# BƯỚC 3: TÍNH HISTOGRAM DÙNG THƯ VIỆN OPENCV
# ==========================================================
# cv.calcHist([ảnh], [kênh], mask, [số rổ], [khoảng giá trị])
hist_opencv = cv.calcHist([img], [0], None, [256], [0, 256])

# ==========================================================
# BƯỚC 4: HIỂN THỊ ĐỒNG THỜI 2 BIỂU ĐỒ ĐỂ QUAN SÁT
# ==========================================================
# Tạo một khung hình lớn chứa 2 biểu đồ con
plt.figure(figsize=(12, 6))
plt.suptitle(f"SO SANH HISTOGRAM - LAB 4\n(Anh: {path})", fontsize=16)

# --- Biểu đồ 1: Cách tính thủ công ---
# plt.subplot(hàng, cột, vị trí)
plt.subplot(1, 2, 1) 
plt.bar(range(256), hist_manual, color='gray', width=1.0)
plt.title("1. Cach tinh THU CONG")
plt.xlabel("Muc sang")
plt.ylabel("So luong pixel")

# --- Biểu đồ 2: Cách dùng OpenCV ---
plt.subplot(1, 2, 2)
plt.plot(hist_opencv, color='blue')
plt.fill_between(range(256), hist_opencv.flatten(), color='blue', alpha=0.3)
plt.title("2. Cach dung thu vien OPENCV")
plt.xlabel("Muc sang")
plt.ylabel("So luong pixel")
plt.xlim([0, 256])

# Tự động căn chỉnh để các tiêu đề không bị đè lên nhau
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()