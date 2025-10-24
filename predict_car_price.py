"""
Script dự đoán giá xe sử dụng mô hình đã huấn luyện
"""

import joblib
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime

# Đường dẫn tới mô hình tốtt nhất
base_dir = os.path.dirname(os.path.abspath(__file__))
candidate_model_paths = [
    os.path.join(base_dir, "data", "processed", "models", "RandomForest_model.pkl"),
]

MODEL_PATH = None
for p in candidate_model_paths:
    if os.path.exists(p):
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    # default to first candidate so error message is clearer in load_model
    MODEL_PATH = candidate_model_paths[0]

def load_model():
    """Nạp mô hình đã huấn luyện"""
    if not os.path.exists(MODEL_PATH):
        print(f"Lỗi: Không tìm thấy file mô hình tại {MODEL_PATH}")
        sys.exit(1)
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Đã nạp mô hình từ {MODEL_PATH}")
        return model
    except Exception as e:
        print(f"Lỗi khi nạp mô hình: {str(e)}")
        sys.exit(1)

def predict_car_price(brand, year, body_type, engine_capacity, fuel_type, is_imported, mileage_km=30000):
    """Dự đoán giá xe dựa trên các thông số đầu vào
    
    Args:
        brand (str): Thương hiệu xe (Mazda, Toyota, Honda, ...)
        year (int): Năm sản xuất
        body_type (str): Kiểu dáng xe (Sedan, SUV, Hatchback, ...)
        engine_capacity (float): Dung tích động cơ (đơn vị: lít)
        fuel_type (str): Loại nhiên liệu (Gasoline, Diesel, Electric, Hybrid)
        is_imported (int): Xe nhập khẩu (1) hay trong nước (0)
        mileage_km (int, optional): Số km đã đi. Mặc định là 30000.
    
    Returns:
        float: Giá dự đoán (đơn vị: triệu VND)
    """
    # Nạp mô hình
    model = load_model()
    
    # Tính tuổi xe
    current_year = datetime.now().year
    car_age = current_year - year
    
    # Tạo DataFrame chứa dữ liệu đầu vào
    input_data = pd.DataFrame({
        'year': [year],
        'engine_capacity': [engine_capacity],
        'car_age': [car_age],
        'is_imported': [is_imported],
        'mileage_km': [mileage_km],
        'brand': [brand],
        'body_type': [body_type],
        'fuel_type': [fuel_type]
    })
    # Ensure categorical columns are strings to match training preprocessing
    for col in ['brand', 'body_type', 'fuel_type', 'mileage_km']:
        if col in input_data.columns:
            input_data[col] = input_data[col].astype(str)
    
    # Dự đoán giá xe
    try:
        predicted_price = model.predict(input_data)[0]
        return predicted_price
    except Exception as e:
        print(f"Lỗi khi dự đoán giá xe: {str(e)}")
        return None

def main():
    print("=== CHƯƠNG TRÌNH DỰ ĐOÁN GIÁ XE ===")
    
    # Sử dụng giá trị mặc định cho demo
    print("Sử dụng các giá trị mẫu:")
    brand = "Honda"
    year = 2020
    body_type = "Sedan" 
    engine_capacity = 1.8
    fuel_type = "Gasoline"
    is_imported = 0
    mileage_km = 30000
    
    print(f"Thương hiệu: {brand}")
    print(f"Năm sản xuất: {year}")
    print(f"Kiểu dáng: {body_type}")
    print(f"Dung tích động cơ: {engine_capacity} lít")
    print(f"Loại nhiên liệu: {fuel_type}")
    print(f"Nhập khẩu: {'Có' if is_imported == 1 else 'Không'}")
    print(f"Số km đã đi: {mileage_km} km")
    
    # Dự đoán giá xe
    predicted_price = predict_car_price(
        brand, year, body_type, engine_capacity, fuel_type, is_imported, mileage_km
    )
    
    if predicted_price is not None:
        print(f"\nGiá xe dự đoán: {predicted_price:.2f} triệu VND")
        if predicted_price >= 1000:
            billions = int(predicted_price // 1000)
            millions = int(predicted_price % 1000)
            print(f"(tương đương: {billions} tỷ {millions} triệu VND)")
    else:
        print("Không thể dự đoán giá xe với thông tin đã cung cấp.")

if __name__ == "__main__":
    main()