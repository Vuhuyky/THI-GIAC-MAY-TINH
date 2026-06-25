import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# ==========================================
# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU TỰ CHỌN (CIFAR-10)
# ==========================================
# Tải bộ ảnh màu kích thước 32x32 pixel của 10 loại đối tượng
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Chuẩn hóa giá trị pixel về khoảng [0, 1] để mạng tính toán ổn định
X_train = X_train / 255.0
X_test = X_test / 255.0

# ==========================================
# 2. XÂY DỰNG MẠNG NƠ-RON ĐA TẦNG (MLP)
# ==========================================
model = models.Sequential([
    # TẦNG ĐẦU VÀO: Ảnh màu 32x32x3 (3 là kênh màu RGB). 
    # Duỗi phẳng ma trận 3D này thành một hàng dọc gồm 32 * 32 * 3 = 3072 con số đầu vào.
    layers.Flatten(input_shape=(32, 32, 3)),
    
    # TẦNG ẨN 1 (Hidden Layer 1): Gồm 512 nơ-ron học các đặc trưng cơ bản (Màu sắc, độ sáng)
    layers.Dense(512, activation='relu'),
    
    # TẦNG ẨN 2 (Hidden Layer 2): Gồm 256 nơ-ron kết hợp dữ liệu từ tầng 1 để học hình dáng phẳng
    layers.Dense(256, activation='relu'),
    
    # TẦNG ẨN 3 (Hidden Layer 3): Gồm 128 nơ-ron đi sâu vào chi tiết trừu tượng của đối tượng
    layers.Dense(128, activation='relu'),
    
    # TẦNG ĐẦU RA (Output Layer): Gồm 10 nơ-ron đại diện cho xác suất của 10 lớp đối tượng
    layers.Dense(10, activation='softmax')
])

# ==========================================
# 3. BIÊN DỊCH VÀ HUẤN LUYỆN MÔ HÌNH
# ==========================================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Cho mạng học qua 10 vòng (epochs) với kích thước mỗi lô dữ liệu là 64 ảnh
print("--- BẮT ĐẦU HUẤN LUYỆN MẠNG MULTI-LAYER PERCEPTRON (MLP) ---")
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

# ==========================================
# 4. ĐÁNH GIÁ VÀ TRỰC QUAN HÓA
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nĐỘ CHÍNH XÁC CỦA MLP TRÊN TẬP KIỂM TRA (CIFAR-10): {test_acc * 100:.2f}%")