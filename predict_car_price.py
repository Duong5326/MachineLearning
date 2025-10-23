"""
Script dự đoán giá xe sử dụng mô hình đã huấn luyện
"""

import joblib
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime

# Đường dẫn đến thư mục chứa mô hình
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "MachineLearning", "src", "models", "Lasso_model.pkl")

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
    
    # Chuyển mileage_km thành dạng phân loại giống khi huấn luyện (binning)
    def mileage_bin(km):
        try:
            if km is None or (isinstance(km, float) and np.isnan(km)):
                return 'unknown'
            km = float(km)
        except Exception:
            return 'unknown'
        if km <= 10000:
            return '0-10k'
        elif km <= 50000:
            return '10-50k'
        elif km <= 100000:
            return '50-100k'
        elif km <= 200000:
            return '100-200k'
        else:
            return '>200k'

    mileage_cat = mileage_bin(mileage_km)

    # Tạo DataFrame chứa dữ liệu đầu vào (bảo đảm các cột categorical là kiểu string)
    input_data = pd.DataFrame({
        'year': [year],
        'engine_capacity': [engine_capacity],
        'car_age': [car_age],
        'is_imported': [is_imported],
        'mileage_km': [mileage_cat],
        'brand': [str(brand)],
        'body_type': [str(body_type)],
        'fuel_type': [str(fuel_type)]
    })
    
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