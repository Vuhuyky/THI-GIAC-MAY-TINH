import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

# ==========================================
# 1. TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU (DATASET)
# ==========================================

# Tải tập dữ liệu Chó và Mèo đã được lọc nhỏ gọn từ Google
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs_filtered.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

BATCH_SIZE = 32
# BẪY TRẮC NGHIỆM: MobileNetV2 bắt buộc kích thước ảnh đầu vào là 160x160
IMG_SIZE = (160, 160)

# Khởi tạo tập dữ liệu Train (Huấn luyện)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# Khởi tạo tập dữ liệu Validation (Kiểm thử)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# Tối ưu hóa hiệu năng tải ảnh bằng cơ chế gối đầu dữ liệu
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


# ==========================================
# 2. ĐỊNH NGHĨA CÁC TẦNG BỔ TRỢ (AUGMENTATION & RESCALING)
# ==========================================

# Tầng tăng cường dữ liệu (Data Augmentation) - Chỉ chạy khi Train
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

# BẪY TRẮC NGHIỆM KINH ĐIỂN: MobileNetV2 yêu cầu pixel nằm trong khoảng [-1, 1]
rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)


# ==========================================
# 3. KHỞI TẠO MÔ HÌNH NỀN (BASE MODEL)
# ==========================================

# Tải mạng MobileNetV2 khổng lồ của Google đã học trên ImageNet
# include_top=False: Loại bỏ tầng phân loại 1000 lớp cũ của nó đi
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)

# ĐÓNG BĂNG MÔ HÌNH NỀN: Ép các trọng số đã có sẵn không bị thay đổi khi train
base_model.trainable = False

# Kiểm tra cấu trúc khối trích xuất đặc trưng của mô hình nền
base_model.summary()


# ==========================================
# 4. THIẾT KẾ PHẦN ĐUÔI VÀ GHÉP MÔ HÌNH
# ==========================================

# GlobalAveragePooling2D: Tính trung bình không gian để nén khối 3D thành vector 1D
# Kỹ thuật này giúp giảm hàng triệu tham số so với tầng Flatten() truyền thống
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# Tầng đầu ra Dense(1) cho bài toán phân loại nhị phân (Chó vs Mèo)
# Không cài activation nghĩa là đầu ra xuất ra số thô (Logits)
prediction_layer = tf.keras.layers.Dense(1)

# Ghép nối tất cả các thành phần lại thành một Model hoàn chỉnh
inputs = tf.keras.Input(shape=(160, 160, 3))
x = data_augmentation(inputs)
x = rescale(x)
# Chú ý training=False giúp các tầng nội bộ như BatchNorm không bị xáo trộn
x = base_model(x, training=False)
x = global_average_layer(x)
outputs = prediction_layer(x)

model = tf.keras.Model(inputs, outputs)


# ==========================================
# 5. BIÊN DỊCH & HUẤN LUYỆN GIAI ĐOẠN 1: FEATURE EXTRACTION
# ==========================================

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), # Phân loại nhị phân + dùng logits thô
    metrics=['accuracy']
)

model.summary() # Lúc này số tham số cần học (Trainable params) cực kỳ ít

# Chạy huấn luyện mô hình nền đóng băng trong 10 Epoch ban đầu
initial_epochs = 10
history = model.fit(
    train_dataset,
    epochs=initial_epochs,
    validation_data=validation_dataset
)


# ==========================================
# 6. GIAI ĐOẠN 2: FINE-TUNING (TINH CHỈNH SÂU)
# ==========================================

# Mở băng toàn bộ mô hình nền ra để chuẩn bị tinh chỉnh các tầng sâu
base_model.trainable = True

# BẪY ĐIỂM 10: Chỉ mở băng các tầng cuối (từ tầng 100 trở đi), đóng băng các tầng đầu
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# BẪY TỐI THƯỢNG: Khi Fine-tuning, bắt buộc phải giảm Learning Rate xuống SIÊU NHỎ (ví dụ chia 10)
# để tránh làm phá hủy hoàn toàn các kiến thức cũ của mô hình gốc (Catastrophic Forgetting)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
    metrics=['accuracy']
)

model.summary() # Số tham số cần học (Trainable params) bây giờ đã tăng lên rất nhiều

# Chạy huấn luyện Fine-tuning thêm 10 Epoch nữa (tổng cộng thành 20 Epoch)
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1], # Tiếp tục mạch Epoch cũ
    validation_data=validation_dataset
)


# ==========================================
# 7. ĐÁNH GIÁ VÀ KIỂM THỬ MÔ HÌNH VỚI ẢNH MỚI
# ==========================================

# Lấy thử một batch ảnh từ tập validation để test dự đoán
image_batch, label_batch = next(iter(validation_dataset))
predictions = model.predict(image_batch)

# Áp dụng hàm kích hoạt Sigmoid vì đây là phân loại nhị phân xuất ra số thô (Logits)
predictions = tf.nn.sigmoid(predictions)
# Nếu xác suất > 0.5 là lớp 1 (Chó), ngược lại là lớp 0 (Mèo)
predictions = tf.where(predictions < 0.5, 0, 1)

print("Kết quả dự đoán của 10 ảnh đầu tiên trong mẻ:")
print(predictions.numpy()[:10].flatten())
print("Nhãn thực tế của 10 ảnh đó:")
print(label_batch.numpy()[:10])