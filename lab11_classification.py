import os
import numpy as np
import cv2 as cv
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. TỰ ĐỘNG TẠO DỮ LIỆU ẢNH MẪU ĐỂ HUẤN LUYỆN
# ==========================================
print("Đang khởi tạo dữ liệu ảnh mẫu cho 2 lớp (Class A và Class B)...")
os.makedirs('dataset/class_A', exist_ok=True)
os.makedirs('dataset/class_B', exist_ok=True)

# Tạo 20 ảnh cho Class A (Chứa các hình tròn)
for i in range(20):
    img = np.zeros((200, 200), dtype=np.uint8)
    for _ in range(5):
        cv.circle(img, (np.random.randint(20, 180), np.random.randint(20, 180)), np.random.randint(10, 40), 255, 2)
    cv.imwrite(f'dataset/class_A/circle_{i}.jpg', img)

# Tạo 20 ảnh cho Class B (Chứa các đường thẳng xoay hướng)
for i in range(20):
    img = np.zeros((200, 200), dtype=np.uint8)
    for _ in range(5):
        pt1 = (np.random.randint(10, 190), np.random.randint(10, 190))
        pt2 = (np.random.randint(10, 190), np.random.randint(10, 190))
        cv.line(img, pt1, pt2, 255, 2)
    cv.imwrite(f'dataset/class_B/line_{i}.jpg', img)

# ==========================================
# 2. TRÍCH XUẤT ĐẶC TRƯNG BẰNG ORB
# ==========================================
print("\n--- Bước 1: Trích xuất đặc trưng ORB từ các ảnh ---")
orb = cv.ORB_create(nfeatures=500)

all_descriptors = []
image_data = [] # Lưu cặp (descriptor của ảnh, nhãn lớp)

for label, class_dir in enumerate(['dataset/class_A', 'dataset/class_B']):
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        img = cv.imread(fpath, cv.IMREAD_GRAYSCALE)
        
        # Tìm điểm đặc trưng và bộ mô tả (descriptors)
        kp, des = orb.detectAndCompute(img, None)
        
        if des is not None:
            all_descriptors.append(des)
            image_data.append((des, label))

# Gộp tất cả descriptor lại thành một ma trận lớn để chạy K-Means
all_descriptors = np.vstack(all_descriptors)

# ==========================================
# 3. XÂY DỰNG TỪ ĐIỂN THỊ GIÁC (VOCABULARY) BẰNG K-MEANS
# ==========================================
print("\n--- Bước 2: Tạo từ điển thị giác (K-Means Clustering) ---")
num_clusters = 20 # Số lượng "từ ngữ thị giác" (K)
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
kmeans.fit(all_descriptors.astype(float))

# ==========================================
# 4. CHUYỂN ĐỔI ẢNH THÀNH BIỂU ĐỒ TẦN SUẤT (HISTOGRAM)
# ==========================================
print("\n--- Bước 3: Biến đổi ảnh thành biểu đồ Bag of Features ---")
X = []
y = []

for des, label in image_data:
    # Dự đoán xem mỗi descriptor thuộc về cụm (từ ngữ) nào
    words = kmeans.predict(des.astype(float))
    
    # Vẽ biểu đồ tần suất xuất hiện của các cụm trong bức ảnh này
    histogram, _ = np.histogram(words, bins=range(num_clusters + 1), density=True)
    
    X.append(histogram)
    y.append(label)

X = np.array(X)
y = np.array(y)

# ==========================================
# ==========================================
# 6. BỔ SUNG: HIỂN THỊ HÌNH ẢNH MINH HỌA ĐỂ CHỤP MÀN HÌNH
# ==========================================
import matplotlib.pyplot as plt

# Đọc thử 1 ảnh ngẫu nhiên để vẽ các điểm đặc trưng ORB lên đó
test_img_path = 'dataset/class_A/circle_0.jpg'
img_sample = cv.imread(test_img_path)
gray_sample = cv.cvtColor(img_sample, cv.COLOR_BGR2GRAY)

# Tìm lại các điểm đặc trưng (Keypoints)
kp, des = orb.detectAndCompute(gray_sample, None)

# Vẽ các vòng tròn màu xanh tại các vị trí đặc trưng mà máy tính nhận diện được
img_keypoints = cv.drawKeypoints(img_sample, kp, None, color=(0, 255, 0), flags=0)

# Hiển thị ảnh kèm theo kết quả dự đoán
plt.figure(figsize=(10, 5))

# Bên trái: Ảnh trích xuất đặc trưng
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB))
plt.title("Cac diem dac trung ORB tim thay")
plt.xticks([]), plt.yticks([])

# Bên phải: Hiển thị bảng biểu đồ phân bổ đặc trưng (Histogram) mẫu của ảnh đó
plt.subplot(1, 2, 2)
words_sample = kmeans.predict(des.astype(float))
hist_sample, _ = np.histogram(words_sample, bins=range(num_clusters + 1), density=True)
plt.bar(range(num_clusters), hist_sample, color='orange', edgecolor='black')
plt.title("Bieu do Bag of Features (Dua vao SVM)")
plt.xlabel("Ma tu vung (Visual Word ID)")
plt.ylabel("Tan suat xuat hien")

print("\nĐang hiển thị hình ảnh trực quan hóa đặc trưng...")
plt.tight_layout()
plt.show()