"""
Script này phân chia dữ liệu đã nâng cao thành tập huấn luyện (train) và tập kiểm tra (test)
"""

import pandas as pd
import os
import sys
import random
import json

# Đường dẫn file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))

# File đầu vào và đầu ra
enhanced_data_file = os.path.join(SCRIPT_DIR, "enhanced_car_data.csv")
train_file = os.path.join(SCRIPT_DIR, "train_data.csv")
test_file = os.path.join(SCRIPT_DIR, "test_data.csv")

def manual_train_test_split(df, test_size=0.2, random_seed=42):
    """
    Tự phân chia dữ liệu thành tập train và test mà không cần scikit-learn
    """
    # Đặt seed ngẫu nhiên để có thể tái tạo kết quả
    random.seed(random_seed)
    
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    df_copy = df.copy()
    
    # Xáo trộn dữ liệu
    indices = list(range(len(df_copy)))
    random.shuffle(indices)
    
    # Tính số lượng mẫu cho tập test
    test_count = int(len(df_copy) * test_size)
    
    # Lấy indices cho tập test và train
    test_indices = indices[:test_count]
    train_indices = indices[test_count:]
    
    # Tạo tập train và test
    test_df = df_copy.iloc[test_indices].reset_index(drop=True)
    train_df = df_copy.iloc[train_indices].reset_index(drop=True)
    
    return train_df, test_df

def split_data(test_size=0.2, random_seed=42):
    """
    Phân chia dữ liệu thành tập train và test
    
    Args:
        test_size (float): Tỷ lệ dữ liệu dành cho tập test (mặc định: 0.2 tức 20%)
        random_seed (int): Số nguyên cho việc tạo ngẫu nhiên (để kết quả có thể tái tạo lại)
    """
    print(f"Đọc dữ liệu đã nâng cao từ {enhanced_data_file}...")
    try:
        # Đọc dữ liệu đã nâng cao
        df = pd.read_csv(enhanced_data_file)
        print(f"Đã đọc {len(df)} mẫu dữ liệu.")
        
        # Kiểm tra dữ liệu
        if len(df) < 5:
            print("CẢNH BÁO: Dữ liệu quá ít để phân chia có ý nghĩa!")
            
        # Phân chia dữ liệu
        train_df, test_df = manual_train_test_split(df, test_size=test_size, random_seed=random_seed)
        
        # Lưu tập train và test
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"Đã phân chia dữ liệu thành:")
        print(f"- Tập huấn luyện (train): {len(train_df)} mẫu -> {train_file}")
        print(f"- Tập kiểm tra (test): {len(test_df)} mẫu -> {test_file}")
        
        # Hiển thị thống kê phân bố dữ liệu
        print("\n=== Thống kê phân bố hãng xe ===")
        print("Tập huấn luyện:")
        print(train_df['brand'].value_counts())
        print("\nTập kiểm tra:")
        print(test_df['brand'].value_counts())
        
        # Thông báo thành công
        print("\nHoàn tất phân chia dữ liệu!")
        return True
        
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return False

if __name__ == "__main__":
    # Kiểm tra xem file dữ liệu đầu vào có tồn tại không
    if not os.path.exists(enhanced_data_file):
        print(f"Lỗi: Không tìm thấy file dữ liệu đã nâng cao: {enhanced_data_file}")
        print("Vui lòng chạy enhance_car_data.py trước khi phân chia dữ liệu!")
        sys.exit(1)
        
    # Phân chia dữ liệu
    split_data()