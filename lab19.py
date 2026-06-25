# =====================================================================
# 1. CÀI ĐẶT CÁC THƯ VIỆN CỐT LÕI
# =====================================================================
!pip install ultralytics roboflow

import os
from ultralytics import YOLO
from roboflow import Roboflow

# =====================================================================
# 2. TẢI DATASET POLYGON (MASK) TỪ ROBOFLOW
# =====================================================================
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY") # Thay thế API Key thực tế của bạn
project = rf.workspace("workspace-name").project("segmentation-project-name")
version = project.version(1)
dataset = version.download("yolov8") # Tải về cấu trúc thư mục YOLOv8


# =====================================================================
# 3. KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH PHÂN ĐOẠN (YOLO-SEG)
# =====================================================================

# BẪY TRẮC NGHIỆM CHÍ MẠNG: 
# Muốn làm bài toán Phân đoạn, bắt buộc phải dùng mô hình có chữ "-seg" ở đuôi.
# 'yolov8n-seg.pt' nghĩa là YOLOv8 phiên bản Nano dành cho Segmentation.
model = YOLO("yolov8n-seg.pt") 

# Tiến hành Huấn luyện (Train)
model.train(
    data=os.path.join(dataset.location, "data.yaml"), # File chứa cấu hình các lớp ảnh
    epochs=20,        # Số chu kỳ học 
    imgsz=640,        # Kích thước ảnh chuẩn 640x640
    batch=16,         # Kích thước mẻ dữ liệu
    device=0          # Sử dụng GPU để tính toán mặt nạ pixel cho nhanh
)


# =====================================================================
# 4. DỰ ĐOÁN VÀ TRÍCH XUẤT MẶT NẠ (INFERENCE / PREDICT)
# =====================================================================
path_to_new_image = "test_image.jpg"

# Chạy dự đoán ảnh mới tinh
results = model.predict(
    source=path_to_new_image, 
    conf=0.25, 
    save=True # Tự động tô màu vật thể, vẽ viền đa giác và lưu vào 'runs/segment/predict'
)

# Khảo sát sâu cấu trúc đầu ra (Thầy có thể hỏi trắc nghiệm nâng cao):
for r in results:
    # Nếu muốn lấy tọa độ các hộp bao (Bounding Boxes) như bài trước
    boxes = r.boxes 
    
    # ĐẶC TRƯNG RIÊNG CỦA SEGMENTATION: Lấy tọa độ các điểm tạo nên mặt nạ đa giác
    masks = r.masks 
    if masks is not None:
        print("Tọa độ pixel của các vùng phân đoạn (mặt nạ):")
        print(masks.xy[0]) # Trả về mảng tọa độ (x, y) bao quanh vật thể thứ nhất