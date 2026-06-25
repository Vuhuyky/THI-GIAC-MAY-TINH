import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist # Sử dụng bộ ảnh quần áo, giày dép mẫu
import matplotlib.pyplot as plt

# ==========================================
# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU ẢNH
# ==========================================
# Bộ dữ liệu gồm 70,000 ảnh xám kích thước 28x28 pixel của 10 loại thời trang
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Chuẩn hóa pixel về khoảng [0, 1] như chúng ta đã học ở bài trước
X_train = X_train / 255.0
X_test = X_test / 255.0

# ==========================================
# 2. XÂY DỰNG MẠNG NƠ-RON NÔNG (SHALLOW NN)
# ==========================================
# Cấu trúc đúng chuẩn bài học: Input -> ĐÚNG 1 Hidden Layer -> Output
model = models.Sequential([
    # TẦNG ĐẦU VÀO (Input Layer): 
    # Duỗi phẳng bức ảnh ảnh 2D (28x28) thành 1 hàng dọc gồm 784 con số (Giống Flatten ở bài trước)
    layers.Flatten(input_shape=(28, 28)),
    
    # TẦNG ẨN DUY NHẤT (The Only Hidden Layer):
    # Gồm 128 tế bào nơ-ron (Perceptron) hoạt động song song để tìm đặc trưng.
    # activation='relu': Hàm kích hoạt giúp mạng có khả năng vẽ đường cong (Phi tuyến tính)
    layers.Dense(128, activation='relu'),
    
    # TẦNG ĐẦU RA (Output Layer):
    # Gồm 10 nơ-ron tương ứng với xác suất của 10 loại quần áo, giày dép khác nhau.
    layers.Dense(10, activation='softmax')
])

# ==========================================
# 3. BIÊN DỊCH VÀ HUẤN LUYỆN
# ==========================================
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Cho mạng học trong 5 vòng (epochs)
print("--- BẮT ĐẦU HUẤN LUYỆN MẠNG NƠ-RON NÔNG ---")
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# ==========================================
# 4. ĐÁNH GIÁ ĐỘ CHÍNH XÁC
# ==========================================
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nĐỘ CHÍNH XÁC TRÊN TẬP KIỂM TRA: {test_acc * 100:.2f}%")