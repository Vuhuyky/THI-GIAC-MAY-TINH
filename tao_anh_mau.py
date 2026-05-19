import numpy as np
import cv2 as cv

print("Đang tự động tạo 10 ảnh lưới hình tròn mẫu...")

# Cấu hình lưới tròn đối xứng 7x6
rows, cols = 6, 7
spacing = 50

for k in range(1, 11):
    # Tạo một bức ảnh nền trắng kích thước 640x480
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Tạo tọa độ 3D cho các tâm hình tròn
    points = []
    for r in range(rows):
        for c in range(cols):
            x = (c - (cols-1)/2) * spacing
            y = (r - (rows-1)/2) * spacing
            points.append([x, y, 0])
    points = np.array(points, dtype=np.float32)
    
    # Giả lập các góc chụp ngẫu nhiên (xoay và tịnh tiến trong không gian 3D)
    rvec = np.array([np.random.uniform(-0.15, 0.15), np.random.uniform(-0.15, 0.15), np.random.uniform(-0.3, 0.3)])
    tvec = np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(350, 400)])
    
    # Ma trận camera giả định và hệ số làm cong ảnh (Distortion)
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)
    dist_coef = np.array([-0.15, 0.05, 0, 0, 0], dtype=np.float32) 
    
    # Chiếu các điểm 3D thành điểm 2D trên ảnh
    img_pts, _ = cv.projectPoints(points, rvec, tvec, K, dist_coef)
    
    # Vẽ các hình tròn màu đen lên nền trắng
    for pt in img_pts:
        x_img, y_img = int(pt[0][0]), int(pt[0][1])
        if 0 <= x_img < 640 and 0 <= y_img < 480:
            cv.circle(img, (x_img, y_img), 10, (0, 0, 0), -1)
            
    # Lưu ảnh lại với tên circle_grid_01.jpg, circle_grid_02.jpg...
    cv.imwrite(f'circle_grid_{k:02d}.jpg', img)

print("Xong! Bạn đã có 10 ảnh từ 'circle_grid_01.jpg' đến 'circle_grid_10.jpg' trong thư mục.")