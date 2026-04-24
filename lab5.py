import cv2 as cv
import numpy as np
import os
import glob

# --- CÁC HÀM XỬ LÝ VÀ TÍNH TOÁN ---

def imread_unicode(path):
    """Đọc ảnh màu từ đường dẫn chứa tiếng Việt có dấu"""
    try:
        nparr = np.fromfile(path, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR) # Đọc ảnh màu (BGR)
        return img
    except Exception as e:
        print(f"Lỗi đọc file: {path} - {e}")
        return None

def compare_raw_data(img1, img2):
    """
    YÊU CẦU 1: So sánh dữ liệu thô (MSE)
    Logic: Tính trung bình bình phương sai lệch giữa từng pixel.
    Giá trị càng nhỏ, ảnh càng giống nhau về mặt vị trí điểm ảnh.
    """
    # Resize ảnh về cùng kích thước ảnh mẫu
    img2_resized = cv.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Tính Mean Squared Error (MSE)
    # Công thức: MSE = (1/N) * sum((I1 - I2)^2)
    err = np.sum((img1.astype("float") - img2_resized.astype("float")) ** 2)
    mse = err / float(img1.shape[0] * img1.shape[1] * img1.shape[2])
    return mse

def get_hsv_histogram(img):
    """
    Tính Histogram trên hệ màu HSV (Tốt hơn ảnh xám)
    H (Hue): Màu sắc, S (Saturation): Độ bão hòa.
    """
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Tính histogram cho cả kênh H và S để bắt màu sắc tốt hơn
    # H có 180 mức, S có 256 mức. Ở đây dùng 50x60 bin để giảm độ phức tạp.
    hist = cv.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv.normalize(hist, hist, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    return hist.flatten()

def get_spatial_hsv_histogram(img, grid=(4, 4)):
    """
    CÁCH KHẮC PHỤC: Chia lưới ảnh (Spatial Histogram)
    Giúp máy tính nhận diện được "Màu đỏ nằm bên trái" hay "Màu xanh nằm bên phải"
    """
    h, w = img.shape[:2]
    dy, dx = h // grid[0], w // grid[1]
    spatial_features = []
    
    for r in range(grid[0]):
        for c in range(grid[1]):
            # Cắt từng ô nhỏ (cell)
            cell = img[r*dy:(r+1)*dy, c*dx:(c+1)*dx]
            # Tính histogram cho từng ô
            hist = get_hsv_histogram(cell)
            spatial_features.append(hist)
            
    return spatial_features

def compare_features(feat1, feat2):
    """Tính khoảng cách L1 (Manhattan distance) giữa hai vector đặc trưng"""
    return np.sum(np.abs(np.array(feat1) - np.array(feat2)))

# --- CHƯƠNG TRÌNH CHÍNH ---

# 1. Cấu hình đường dẫn
path_query = r'D:\THỊ GIÁC MÁY TÍNH\query.jpg'  
path_database = r'D:\THỊ GIÁC MÁY TÍNH\DATABASE'

# 2. Xử lý ảnh mẫu (Query)
img_q = imread_unicode(path_query)
if img_q is None:
    print("Không tìm thấy ảnh mẫu!")
    exit()

# Trích xuất đặc trưng của ảnh mẫu một lần duy nhất
hist_q_global = get_hsv_histogram(img_q)
hist_q_spatial = get_spatial_hsv_histogram(img_q)

# 3. Quét kho ảnh
extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
files = []
for ext in extensions:
    files.extend(glob.glob(os.path.join(path_database, ext)))
files = list(set(files)) # Loại bỏ trùng lặp

results = []
print(f"Đang phân tích {len(files)} ảnh trong Database...")

for f in files:
    img_db = imread_unicode(f)
    if img_db is None: continue
    
    # Tính toán 3 phương pháp
    # MSE (Raw data)
    d_raw = compare_raw_data(img_q, img_db)
    
    # Global Histogram
    hist_db_global = get_hsv_histogram(img_db)
    d_global = compare_features(hist_q_global, hist_db_global)
    
    # Spatial Histogram (Phương pháp cải tiến)
    hist_db_spatial = get_spatial_hsv_histogram(img_db)
    d_spatial = sum(compare_features(h1, h2) for h1, h2 in zip(hist_q_spatial, hist_db_spatial))
    
    results.append({
        'name': os.path.basename(f),
        'raw': d_raw,
        'global': d_global,
        'spatial': d_spatial
    })

# 4. Sắp xếp kết quả theo Spatial Hist (Vì đây là phương pháp tối ưu nhất trong 3 cái)
results.sort(key=lambda x: x['spatial'])

# 5. Hiển thị kết quả
print("-" * 85)
print(f"{'Tên Ảnh':<25} | {'Raw MSE':<15} | {'Global Hist':<15} | {'Spatial Hist'}")
print("-" * 85)
for res in results[:5]: # Hiển thị Top 5
    print(f"{res['name']:<25} | {res['raw']:<15.2f} | {res['global']:<15.4f} | {res['spatial']:.4f}")

print("-" * 85)
if results:
    print(f"==> KẾT QUẢ: Ảnh giống nhất là '{results[0]['name']}' dựa trên Spatial Histogram.")