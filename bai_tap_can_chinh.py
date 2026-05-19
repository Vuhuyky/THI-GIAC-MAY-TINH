import numpy as np
import cv2 as cv
import glob

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ LƯỚI HÌNH TRÒN
# ==========================================
pattern_size = (7, 6) # Lưới gồm 7 cột và 6 hàng chấm tròn

# Chuẩn bị tọa độ các điểm trong thế giới thực (Object Points)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

objpoints = [] # Lưu điểm 3D thực tế
imgpoints = [] # Lưu vị trí tâm hình tròn 2D trên ảnh

# Đọc tất cả các ảnh mẫu có tên bắt đầu bằng circle_grid_
images = glob.glob('circle_grid_*.jpg') 

print(f"Tìm thấy {len(images)} ảnh để xử lý.")

# ==========================================
# 2. TÌM TÂM CÁC HÌNH TRÒN (IMAGE POINTS)
# ==========================================
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # DÙNG HÀM TÌM LƯỚI TRÒN ĐỐI XỨNG THEO YÊU CẦU ĐỀ BÀI
    ret, centers = cv.findCirclesGrid(gray, pattern_size, cv.CALIB_CB_SYMMETRIC_GRID)

    if ret == True:
        objpoints.append(objp)
        imgpoints.append(centers)

        # Vẽ các tâm hình tròn tìm được lên ảnh để hiển thị
        cv.drawChessboardCorners(img, pattern_size, centers, ret)
        cv.imshow('Dang tim quĩ dao tam tron...', img)
        cv.waitKey(300) # Đợi 0.3 giây mỗi ảnh để bạn quan sát
    else:
        print(f"Không tìm thấy lưới tròn trong ảnh: {fname}")

cv.destroyAllWindows()

# ==========================================
# 3. TIẾN HÀNH CÂN CHỈNH CAMERA
# ==========================================
print("\n--- Đang tính toán ma trận Camera... ---")
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("\n[KẾT QUẢ VỀ TÍNH CHẤT NỘI TẠI - INTRINSIC]")
print("Ma trận Camera (Camera Matrix):\n", mtx)
print("Hệ số làm cong/méo ảnh (Distortion Coefficients):\n", dist)

# ==========================================
# 4. KHỬ BIẾN DẠNG ẢNH (UNDISTORT)
# ==========================================
# Lấy thử ảnh số 1 ra để nắn thẳng lại
img_test = cv.imread('circle_grid_01.jpg')
h, w = img_test.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# Tiến hành nắn thẳng ảnh
dst = cv.undistort(img_test, mtx, dist, None, newcameramtx)

# Cắt bỏ rìa đen thừa nếu có
x, y, w_box, h_box = roi
dst = dst[y:y+h_box, x:x+w_box]

# Lưu lại ảnh kết quả sạch sẽ
cv.imwrite('ket_qua_khu_cong.png', dst)
print("\nĐã nắn thẳng ảnh và lưu thành công tại file: 'ket_qua_khu_cong.png'")

# ==========================================
# 5. TÍNH SAI SỐ CHIẾU LẠI (RE-PROJECTION ERROR)
# ==========================================
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2SQR) / len(imgpoints2)
    mean_error += error

print("\nSai số tổng kết (Re-projection Error): {}".format(np.sqrt(mean_error / len(objpoints))))
print("-> Sai số càng gần bằng 0 tức là thuật toán tính càng chính xác!")