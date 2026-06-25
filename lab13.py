# ==========================================
# 1. KHỞI TẠO VÀ CHUẨN BỊ DỮ LIỆU
# ==========================================
# Tải bộ dữ liệu mẫu Iris (Dữ liệu về 3 loài hoa lan dựa trên kích thước hoa)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Tải dữ liệu vào biến iris
iris = load_iris()

# X chứa các đặc trưng đầu vào (chiều dài, chiều rộng của cánh hoa)
X = iris.data 
# y chứa nhãn của hoa (loại 0, loại 1, loại 2)
y = iris.target 

# Chia dữ liệu: 80% để học (Train), 20% để làm bài kiểm tra (Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. CHUẨN HÓA DỮ LIỆU (YẾU TỐ QUYẾT ĐỊNH)
# ==========================================
sc = StandardScaler()
# Máy tính tính toán giá trị trung bình và độ lệch chuẩn của tập Train
sc.fit(X_train)
# Ép dữ liệu về dạng chuẩn có trung bình bằng 0 và độ lệch chuẩn bằng 1
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
# ==========================================
# 3. KHỞI TẠO VÀ HUẤN LUYỆN PERCEPTRON
# ==========================================
# Tạo ra một tế bào nơ-ron Perceptron
# eta0: Tốc độ học (Learning rate) - bước đi của máy tính mỗi lần sửa sai
# max_iter: Số vòng lặp tối đa qua toàn bộ dữ liệu (giống epoch trong CNN)
p = Perceptron(max_iter=40, eta0=0.1, random_state=42)

# Bắt đầu cho Perceptron học dữ liệu
p.fit(X_train_std, y_train)
# ==========================================
# 4. ĐÁNH GIÁ ĐỘ CHÍNH XÁC
# ==========================================
# Cho Perceptron làm bài kiểm tra trên tập dữ liệu Test chưa từng được thấy
y_pred = p.predict(X_test_std)

# Tính toán xem dự đoán đúng được bao nhiêu phần trăm
print(f"ĐỘ CHÍNH XÁC CỦA PERCEPTRON: {accuracy_score(y_test, y_pred) * 100:.2f}%")