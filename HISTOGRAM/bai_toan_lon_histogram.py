import cv2 as cv
import numpy as np
import os
import glob
from send2trash import send2trash  # <--- Khai báo thư viện ném vào thùng rác

# ==========================================
# HÀM 1: TÍNH VÀ CHUẨN HÓA HISTOGRAM THỦ CÔNG
# ==========================================
def tinh_histogram_thu_cong(anh_xam):
    H = [0] * 256
    h, w = anh_xam.shape
    N = h * w
    
    for i in range(h):
        for j in range(w):
            muc_sang = anh_xam[i, j] 
            H[muc_sang] += 1         
            
    H_norm = [0.0] * 256
    for k in range(256):
        H_norm[k] = H[k] / N
        
    return H_norm

# ==========================================
# HÀM 2: TÍNH KHOẢNG CÁCH d THỦ CÔNG (Manhattan)
# ==========================================
def tinh_khoang_cach_thu_cong(hist1, hist2):
    d = 0.0
    for i in range(256):
        d += abs(hist1[i] - hist2[i])
    return d

# ==========================================
# HÀM 3: TÌM VÀ NÉM ẢNH TRÙNG LẶP VÀO THÙNG RÁC
# ==========================================
def tim_anh_trung_lap(duong_dan_thu_muc, nguong_sai_so=0.05):
    print(f"Đang quét thư mục: {duong_dan_thu_muc}...\n")
    
    danh_sach_file = glob.glob(os.path.join(duong_dan_thu_muc, "*.jpg"))
    so_luong_anh = len(danh_sach_file)
    
    if so_luong_anh < 2:
        print("Cần ít nhất 2 ảnh để so sánh!")
        return

    print("Đang phân tích dữ liệu ảnh, vui lòng đợi...")
    du_lieu_hist = {}
    for file_anh in danh_sach_file:
        anh_goc = cv.imread(file_anh)
        if anh_goc is not None:
            anh_xam = cv.cvtColor(anh_goc, cv.COLOR_BGR2GRAY)
            du_lieu_hist[file_anh] = tinh_histogram_thu_cong(anh_xam)

    print("Hoàn tất phân tích. Bắt đầu so sánh...\n")
    print("-" * 50)

    danh_sach_da_xoa = [] 
    
    for i in range(so_luong_anh - 1):
        anh_A = danh_sach_file[i]
        
        if anh_A in danh_sach_da_xoa:
            continue
            
        for j in range(i + 1, so_luong_anh):
            anh_B = danh_sach_file[j]
            
            if anh_B in danh_sach_da_xoa:
                continue
                
            hist_A = du_lieu_hist[anh_A]
            hist_B = du_lieu_hist[anh_B]
            
            khoang_cach = tinh_khoang_cach_thu_cong(hist_A, hist_B)
            
            if khoang_cach <= nguong_sai_so:
                print(f"\n[PHÁT HIỆN TRÙNG LẶP]")
                print(f" - Ảnh gốc: {os.path.basename(anh_A)}")
                print(f" - Ảnh nghi trùng: {os.path.basename(anh_B)}")
                print(f" -> Độ lệch (d): {khoang_cach:.4f}")
                
                while True:
                    lua_chon = input(f"   => Bạn có muốn ném ảnh '{os.path.basename(anh_B)}' vào THÙNG RÁC không? (y/n): ").strip().lower()
                    
                    if lua_chon in ['y', 'yes']:
                        # THAY ĐỔI QUAN TRỌNG Ở ĐÂY: Dùng send2trash thay vì os.remove
                        try:
                            send2trash(anh_B)
                            print(f"   [OK] Đã chuyển vào Thùng rác: {os.path.basename(anh_B)}")
                            danh_sach_da_xoa.append(anh_B)
                        except Exception as e:
                            print(f"   [LỖI] Không thể ném vào thùng rác. Lỗi: {e}")
                        break 
                        
                    elif lua_chon in ['n', 'no']:
                        print(f"   [SKIP] Đã giữ lại ảnh: {os.path.basename(anh_B)}")
                        break 
                        
                    else:
                        print("   [LỖI] Vui lòng chỉ nhập 'y' (để ném thùng rác) hoặc 'n' (để bỏ qua)!")

# ==========================================
# CHẠY CHƯƠNG TRÌNH
# ==========================================
thu_muc_cua_ban = r"E:\KYYYYYYY\KY" 

tim_anh_trung_lap(thu_muc_cua_ban, nguong_sai_so=0.05)
print("\n" + "="*50)
print("Hoàn tất quá trình dọn dẹp thư mục!")