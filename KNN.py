import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# =========================
# TẠO DATASET
# =========================

X = []
y = []

# =========================
# ĐỌC ẢNH GÀ
# =========================

ga_path = "GA"

for file in os.listdir(ga_path):

    img_path = os.path.join(ga_path, file)

    img = cv2.imread(img_path)

    # kiểm tra ảnh lỗi
    if img is None:
        continue

    # resize ảnh
    img = cv2.resize(img, (100, 100))

    # tính histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # chuyển thành vector
    hist = hist.flatten()

    # thêm vào dataset
    X.append(hist)

    # label 0 = gà
    y.append(0)

# =========================
# ĐỌC ẢNH VỊT
# =========================

vit_path = "VIT"

for file in os.listdir(vit_path):

    img_path = os.path.join(vit_path, file)

    img = cv2.imread(img_path)

    # kiểm tra ảnh lỗi
    if img is None:
        continue

    # resize ảnh
    img = cv2.resize(img, (100, 100))

    # tính histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # chuyển thành vector
    hist = hist.flatten()

    # thêm vào dataset
    X.append(hist)

    # label 1 = vịt
    y.append(1)

# =========================
# CHUYỂN SANG NUMPY ARRAY
# =========================

X = np.array(X)
y = np.array(y)

# =========================
# TRAIN KNN
# =========================

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X, y)

print("Train xong!")

# =========================
# TEST ẢNH MỚI
# =========================

test_img = cv2.imread("query_KNN2.jpg")

if test_img is None:
    print("Không tìm thấy query_KNN2.jpg")
    exit()

# resize
test_img = cv2.resize(test_img, (100, 100))

# histogram
test_hist = cv2.calcHist([test_img], [0], None, [256], [0, 256])

# vector
test_hist = test_hist.flatten()

# predict
result = knn.predict([test_hist])

# =========================
# HIỂN THỊ KẾT QUẢ
# =========================

if result[0] == 0:
    print("Đây là GÀ")
else:
    print("Đây là VỊT")

# hiển thị ảnh
cv2.imshow("Query Image", test_img)

cv2.waitKey(0)
cv2.destroyAllWindows()