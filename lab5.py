import cv2 as cv
import numpy as np
import os
import glob

# --- CÁC HÀM XỬ LÝ VÀ TÍNH TOÁN ---

def imread_unicode(path, flags=cv.IMREAD_GRAYSCALE):
    """Đọc ảnh từ đường dẫn chứa tiếng Việt có dấu"""
    try:
        nparr = np.fromfile(path, np.uint8)
        img = cv.imdecode(nparr, flags)
        return img
    except Exception as e:
        print(f"Lỗi đọc file: {path} - {e}")
        return None

def compare_raw_data(img1, img2):
    """YÊU CẦU 1: So sánh dữ liệu thô (MSE)"""
    # Resize ảnh về cùng kích thước ảnh mẫu để trừ pixel
    img2_resized = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    # Tính sai số bình phương trung bình (Mean Squared Error)
    err = np.sum((img1.astype("float") - img2_resized.astype("float")) ** 2)
    return err / float(img1.shape[0] * img1.shape[1])

def get_hist_norm(img):
    """Tính Histogram toàn cục và chuẩn hóa"""
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    return hist / np.sum(hist)

def compare_histogram(hist1, hist2):
    """YÊU CẦU 2: Khoảng cách Histogram toàn cục (L1 Distance)"""
    return np.sum(np.abs(hist1 - hist2))

def get_spatial_histogram(img, grid=(4, 4)):
    """CÁCH KHẮC PHỤC: Chia lưới ảnh (Spatial Histogram)"""
    h, w = img.shape
    dy, dx = h // grid[0], w // grid[1]
    spatial_hist = []
    for r in range(grid[0]):
        for c in range(grid[1]):
            # Cắt từng ô nhỏ (cell) trong lưới
            cell = img[r*dy:(r+1)*dy, c*dx:(c+1)*dx]
            hist = cv.calcHist([cell], [0], None, [256], [0, 256])
            sum_h = np.sum(hist)
            spatial_hist.append(hist / sum_h if sum_h > 0 else hist)
    return spatial_hist

def compare_spatial_histogram(shist1, shist2):
    """Tính tổng khoảng cách của tất cả các ô trong lưới"""
    return sum(np.sum(np.abs(h1 - h2)) for h1, h2 in zip(shist1, shist2))

# --- CHƯƠNG TRÌNH CHÍNH ---

# 1. Đường dẫn thực tế trên máy của bạn
path_query = r'D:\THỊ GIÁC MÁY TÍNH\query.jpg'
path_database = r'D:\THỊ GIÁC MÁY TÍNH\DATABASE'

# 2. Xử lý ảnh mẫu
img_q = imread_unicode(path_query, cv.IMREAD_GRAYSCALE)
if img_q is None:
    print("Không thể tải ảnh mẫu. Vui lòng kiểm tra lại đường dẫn!")
    exit()

hist_q = get_hist_norm(img_q)
sh_q = get_spatial_histogram(img_q)

# 3. Quét kho ảnh (Dùng set để tránh lặp file do Windows hoa/thường)
all_files = glob.glob(os.path.join(path_database, "*.[jJ][pP][gG]")) + \
            glob.glob(os.path.join(path_database, "*.png"))
files = list(set(all_files))

results = []
print(f"Đang thực hiện truy vấn trên {len(files)} ảnh...")

for f in files:
    img_db = imread_unicode(f, cv.IMREAD_GRAYSCALE)
    if img_db is None: continue
    
    # Thực hiện 3 phương pháp so sánh
    d_raw = compare_raw_data(img_q, img_db)
    d_global = compare_histogram(hist_q, get_hist_norm(img_db))
    d_spatial = compare_spatial_histogram(sh_q, get_spatial_histogram(img_db))
    
    results.append({
        'name': os.path.basename(f),
        'raw': d_raw,
        'global': d_global,
        'spatial': d_spatial
    })

# 4. Sắp xếp kết quả (Ưu tiên Global Dist để tìm ảnh cùng bộ ảnh mẫu)
# Bạn có thể đổi sang 'spatial' nếu muốn ưu tiên vị trí mảng màu
results.sort(key=lambda x: x['global'])

# 5. Hiển thị kết quả ra bảng
print("-" * 75)
print(f"{'Tên Ảnh':<20} | {'Raw MSE':<12} | {'Global Hist':<12} | {'Spatial Hist'}")
print("-" * 75)
for res in results[:5]: # Hiển thị Top 5 kết quả tốt nhất
    print(f"{res['name']:<20} | {res['raw']:<12.0f} | {res['global']:<12.4f} | {res['spatial']:.4f}")

print("-" * 75)
print(f"==> KET QUA: Anh '{results[0]['name']}' co do tuong dong cao nhat.")