# =====================================================================
# 1. CÀI ĐẶT THƯ VIỆN CẦN THIẾT
# =====================================================================
# Cài đặt thư viện ultralytics (chứa mô hình YOLO) và roboflow (quản lý dữ liệu)
!pip install ultralytics roboflow

import os
from ultralytics import YOLO
from roboflow import Roboflow

# =====================================================================
# 2. TẢI DATASET ĐÃ GÁN NHÃN TỪ ROBOFLOW
# =====================================================================
# Thầy sẽ lấy đoạn mã này trực tiếp trên Roboflow sau khi bạn Export dữ liệu.
# Bạn cần thay thế API_KEY và tên dự án bằng thông tin thực tế của bạn.

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY") # Thay API Key của bạn vào đây
project = rf.workspace("workspace-name").project("project-name")
version = project.version(1)
dataset = version.download("yolov8") # Tải về dưới định dạng cấu trúc của YOLOv8

# Sau khi chạy lệnh này, một thư mục chứa dữ liệu và file `data.yaml` 
# (chứa đường dẫn ảnh + tên các lớp) sẽ được tự động tải về máy.


# =====================================================================
# 3. KHỞI TẠO VÀ HUẤN LUYỆN MÔ HÌNH YOLOv8
# =====================================================================

# BẪY TRẮC NGHIỆM: Chọn kích cỡ mô hình nền (Pre-trained Model)
# 'yolov8n.pt' là bản Nano - Siêu nhẹ, chạy nhanh, phù hợp cho bài tập/thi cử.
# Nếu muốn chính xác cao nhất chấp nhận chạy chậm, thay bằng 'yolov8x.pt' (Extra Large).
model = YOLO("yolov8n.pt") 

# Tiến hành Huấn luyện (Train)
results = model.train(
    data=os.path.join(dataset.location, "data.yaml"), # Đường dẫn tới file cấu hình dữ liệu
    epochs=25,       # Số chu kỳ học (khi đi thi thường để 20-50 để chạy cho nhanh)
    imgsz=640,       # Kích thước không gian ảnh bắt buộc của YOLO (640x640 pixel)
    batch=16,        # Kích thước mẻ dữ liệu (Batch size)
    device=0         # Ép mô hình sử dụng GPU (Card đồ họa) để tăng tốc tính toán
)


# =====================================================================
# 4. DỰ ĐOÁN ẢNH MỚI (INFERENCE / PREDICT)
# =====================================================================

# Đường dẫn tới bức ảnh mới tinh mà mô hình chưa từng được học
path_to_new_image = "test_image.jpg"

# BẪY TRẮC NGHIỆM: Ngưỡng lọc độ tự tin (conf)
# conf=0.25 nghĩa là chỉ hiển thị các ô bao (Bounding Box) có độ tự tin trên 25%.
predictions = model.predict(
    source=path_to_new_image, 
    conf=0.25, 
    save=True # Tự động vẽ ô vuông, ghi tên lớp lên ảnh và lưu lại trong thư mục 'runs/detect/predict'
)


# =====================================================================
# 5. XUẤT MÔ HÌNH RA FILE ĐỂ SỬ DỤNG LÂU DÀI
# =====================================================================
# Sau khi train xong, trọng số tối ưu nhất sẽ được lưu ở file 'runs/detect/train/weights/best.pt'.
# Chúng ta có thể xuất mô hình này sang định dạng khác nếu cần (ví dụ ONNX hoặc TFLite).

path_to_best_weights = "runs/detect/train/weights/best.pt"
trained_model = YOLO(path_to_best_weights)

# Xuất mô hình sang dạng ONNX để chạy mượt mà trên CPU hoặc các ứng dụng C++
trained_model.export(format="onnx")