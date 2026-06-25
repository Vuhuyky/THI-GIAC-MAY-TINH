import os
import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================================
# 1. ĐỊNH NGHĨA CÁC THAM SỐ CẤU TRÚC VIDEO
# =====================================================================
NUM_FRAMES = 10       # Số lượng khung hình trích xuất từ 1 video
IMG_SIZE = 224        # Kích thước chiều dài/rộng của mỗi khung hình
NUM_CLASSES = 10      # Giả sử chúng ta phân loại 10 nhóm hành động thể thao

# Kích thước đầu vào dạng 5D (Không tính Batch Size ở chiều đầu tiên)
# Shape thực tế: (NUM_FRAMES, IMG_SIZE, IMG_SIZE, CHANNELS)
input_shape = (NUM_FRAMES, IMG_SIZE, IMG_SIZE, 3)


# =====================================================================
# 2. XÂY DỰNG KIẾN TRÚC LAI (CNN + RNN / GRU)
# =====================================================================

# Bước A: Tạo một mạng CNN nhỏ làm nhiệm vụ đọc ảnh thô
def build_feature_extractor():
    feature_extractor = tf.keras.Sequential([
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D() # Nén ảnh thành vector đặc trưng 1D
    ])
    return feature_extractor

feature_extractor = build_feature_extractor()

# Bước B: Xây dựng mô hình phân loại Video tổng thể
video_input = layers.Input(shape=input_shape)

# BẪY TRẮC NGHIỆM CHÍ MẠNG: Tầng TimeDistributed
# Nếu không có tầng này, mạng CNN sẽ báo lỗi vì nó không tự hiểu được chiều thời gian (NUM_FRAMES).
# TimeDistributed ép feature_extractor chạy lặp lại trên từng khung hình một của video.
x = layers.TimeDistributed(feature_extractor)(video_input)

# Bước C: Đưa chuỗi đặc trưng thời gian qua tầng GRU (hoặc LSTM) để học hành động
x = layers.GRU(64, return_sequences=False)(x)

# Bước D: Tầng phân loại đầu ra
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(video_input, outputs)


# =====================================================================
# 3. BIÊN DỊCH VÀ KIỂM TRA CẤU TRÚC MẠNG
# =====================================================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Hiển thị bảng tóm tắt kiến trúc mạng để kiểm tra Shape dòng dữ liệu
model.summary()