import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10 # Sử dụng dữ liệu ảnh màu làm mẫu phân loại
import matplotlib.pyplot as plt

# ==========================================
# GIAI ĐOẠN 1: TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================
# Bộ dữ liệu CIFAR-10 gồm 10 lớp ảnh màu kích thước 32x32 pixel
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Chuẩn hóa giá trị pixel về khoảng [0, 1] để tối ưu toán học (như các bài trước)
X_train, X_test = X_train / 255.0, X_test / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ==========================================
# GIAI ĐOẠN 2: THIẾT KẾ KIẾN TRÚC CNN (TỰ CHỌN TẦNG)
# ==========================================
# Thiết kế mạng theo mô hình tuần tự (Sequential) gồm các khối Tích chập lặp lại
model = models.Sequential()

# Khối tích chập 1: 32 bộ lọc kích thước 3x3. Đầu vào là ảnh 32x32x3 (3 kênh màu RGB)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2))) # Thu nhỏ ảnh đi 1 nửa kích thước

# Khối tích chập 2: Tăng lên 64 bộ lọc để học các đặc trưng phức tạp hơn
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Khối tích chập 3: 64 bộ lọc để làm mịn và trích xuất đặc trưng sâu sắc
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# ---- PHẦN PHÂN LOẠI (CLASSIFICATION HEAD) ----
# Duỗi phẳng ma trận đặc trưng 2D thành một hàng dọc các con số (Vector)
model.add(layers.Flatten())
# Tầng ẩn kết nối dày đặc gồm 64 nơ-ron để phân tích dữ liệu
model.add(layers.Dense(64, activation='relu'))
# Tầng đầu ra gồm 10 nơ-ron tương ứng với xác suất của 10 lớp ảnh
model.add(layers.Dense(10))

# ==========================================
# GIAI ĐOẠN 3: BIÊN DỊCH VÀ HUẤN LUYỆN
# ==========================================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print("--- BẮT ĐẦU QUÁ TRÌNH HUẤN LUYỆN CNN TỰ THIẾT KẾ ---")
# Huấn luyện trong 10 vòng (epochs)
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_test, y_test))

# ==========================================
# GIAI ĐOẠN 4: ĐÁNH GIÁ VÀ TRỰC QUAN HÓA
# ==========================================
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(f"\nĐỘ CHÍNH XÁC TRÊN TẬP KIỂM TRA: {test_acc * 100:.2f}%")

# Vẽ đồ thị biểu diễn độ chính xác qua các vòng lặp để nộp báo cáo
plt.plot(history.history['accuracy'], label='Accuracy (Train)')
plt.plot(history.history['val_accuracy'], label = 'Accuracy (Validation)')
plt.xlabel('Vòng lặp (Epoch)')
plt.ylabel('Độ chính xác')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('Bieu do do chinh xac qua 10 epochs - Lab 16')
plt.show()