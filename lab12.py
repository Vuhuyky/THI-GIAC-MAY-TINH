import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

# ==========================================
# 1. TẢI VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================
# Tải bộ dữ liệu ảnh các loài hoa từ Google về máy
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

# Chia 80% dữ liệu để huấn luyện (Train)
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, validation_split=0.2, subset="training", seed=123,
  image_size=(img_height, img_width), batch_size=batch_size)

# Chia 20% dữ liệu để kiểm tra (Validation)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir, validation_split=0.2, subset="validation", seed=123,
  image_size=(img_height, img_width), batch_size=batch_size)

# Tối ưu hóa bộ nhớ đệm giúp máy tính load ảnh mượt hơn, tránh nghẽn cổ chai
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 2. XÂY DỰNG MẠNG NƠ-RON TÍCH CHẬP (CNN MODEL)
# ==========================================
num_classes = 5 # Có 5 loại hoa: Daisy, Dandelion, Roses, Sunflowers, Tulips

model = Sequential([
  # Kỹ thuật Data Augmentation: Tự động xoay, lật, phóng to ảnh ngẫu nhiên lúc học
  # Mục đích: Làm giàu dữ liệu, giúp máy tính không bị "học vẹt" (Overfitting)
  layers.RandomFlip("horizontal", input_shape=(img_height, img_width, 3)),
  layers.RandomRotation(0.1),
  layers.RandomZoom(0.1),
  
  # Chuẩn hóa: Đổi giá trị pixel từ [0-255] về khoảng [0-1] để toán học tính toán nhanh hơn
  layers.Rescaling(1./255),
  
  # Tầng Tích chập 1 (Conv2D): Máy tính dùng 16 ô quét (Kernel) tự động tìm các cạnh, nét của hoa (Giống Bài 2 Sobel)
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  # Tầng Pooling 1: Giảm kích thước ảnh đi một nửa để giữ lại đặc trưng mạnh nhất và tăng tốc độ (Giống việc thu nhỏ ảnh)
  layers.MaxPooling2D(),
  
  # Tầng Tích chập 2: Tăng lên 32 ô quét để tìm các cấu trúc phức tạp hơn (như đường cong cánh hoa)
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Tầng Tích chập 3: Tăng lên 64 ô quét để học các chi tiết sâu sắc tinh vi hơn nữa
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  
  # Tầng Dropout: Tắt ngẫu nhiên 20% nơ-ron trong quá trình học để ép mô hình phải tự tư duy, chống học vẹt
  layers.Dropout(0.2),
  
  # Phẳng hóa: Duỗi phẳng ma trận ảnh 2D thành một hàng dọc các con số (Vector) giống như bước Bag of Features
  layers.Flatten(),
  
  # Tầng kết nối dày đặc (Dense): Nơi phân tích các con số đặc trưng để đưa ra quyết định
  layers.Dense(128, activation='relu'),
  
  # Tầng đầu ra: Trả về xác suất của 5 loài hoa
  layers.Dense(num_classes)
])

# ==========================================
# 3. BIÊN DỊCH VÀ HUẤN LUYỆN MÔ HÌNH
# ==========================================
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Bắt đầu cho máy học trong 15 vòng (epochs)
epochs = 15
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)

# ==========================================
# 4. TRỰC QUAN HÓA KẾT QUẢ HỌC TẬP
# ==========================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Do chinh xac qua cac vong hoc')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Sai so qua cac vong hoc')
plt.show()