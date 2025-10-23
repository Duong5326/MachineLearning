"""
Script tổng hợp để chạy toàn bộ quy trình xử lý dữ liệu xe:
1. Làm sạch dữ liệu từ raw sang processed
2. Nâng cao dữ liệu trong processed
3. Phân chia dữ liệu thành tập huấn luyện (train) và tập kiểm tra (test)
"""

import os
import sys
import subprocess
from pathlib import Path

# Thiết lập đường dẫn
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir  # MachineLearning folder
processed_dir = project_root / "data" / "processed"

# Đường dẫn đến các script
clean_script = processed_dir / "clean_raw_to_processed.py"
enhance_script = processed_dir / "enhance_car_data.py"
split_script = processed_dir / "split_train_test.py"

print("=== QUY TRÌNH XỬ LÝ DỮ LIỆU XE ===")

# Bước 1: Làm sạch dữ liệu từ raw sang processed
print("\n1. Làm sạch dữ liệu từ raw sang processed")
try:
    print(f"Chạy script: {clean_script}")
    result = subprocess.run([sys.executable, str(clean_script)], check=True)
    if result.returncode == 0:
        print("Làm sạch dữ liệu thành công!")
    else:
        print(f"Làm sạch dữ liệu thất bại với mã lỗi: {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"Lỗi khi làm sạch dữ liệu: {e}")
    sys.exit(1)

# Bước 2: Nâng cao dữ liệu
print("\n2. Nâng cao dữ liệu")
try:
    print(f"Chạy script: {enhance_script}")
    result = subprocess.run([sys.executable, str(enhance_script)], check=True)
    if result.returncode == 0:
        print("Nâng cao dữ liệu thành công!")
    else:
        print(f"Nâng cao dữ liệu thất bại với mã lỗi: {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"Lỗi khi nâng cao dữ liệu: {e}")
    sys.exit(1)

# Bước 3: Phân chia dữ liệu thành tập train và test
print("\n3. Phân chia dữ liệu thành tập train và test")
try:
    print(f"Chạy script: {split_script}")
    result = subprocess.run([sys.executable, str(split_script)], check=True)
    if result.returncode == 0:
        print("Phân chia dữ liệu thành công!")
    else:
        print(f"Phân chia dữ liệu thất bại với mã lỗi: {result.returncode}")
        sys.exit(1)
except Exception as e:
    print(f"Lỗi khi phân chia dữ liệu: {e}")
    sys.exit(1)

print("\n=== QUY TRÌNH HOÀN TẤT ===")
print("Dữ liệu đã được xử lý thành công!")
print(f"Các file kết quả:")
print(f"- Dữ liệu đã làm sạch: {processed_dir}/car_data.json")
print(f"- Dữ liệu tiếng Anh: {processed_dir}/car_data_en.json")
print(f"- Dữ liệu đã nâng cao: {processed_dir}/enhanced_car_data.json") 
print(f"- CSV đã nâng cao: {processed_dir}/enhanced_car_data.csv")
print(f"- Tập huấn luyện: {processed_dir}/train_data.csv")
print(f"- Tập kiểm tra: {processed_dir}/test_data.csv")