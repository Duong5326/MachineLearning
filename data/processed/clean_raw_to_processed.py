"""Làm sạch dữ liệu xe từ raw sang processed."""

import json
import os
import pandas as pd
from pathlib import Path
import re

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent.parent
processed_dir = project_root / "data" / "processed"

# Tìm file CSV
candidate_paths = [
    project_root / "data" / "raw" / "used_cars.csv",
    project_root / "data" / "raw" / "data" / "raw" / "used_cars.csv",
    Path(r"d:\New folder\MachineLearning\data\raw\data\raw\used_cars.csv"),
]

used_cars_path = next((p for p in candidate_paths if p.exists()), None)
if not used_cars_path:
    raise FileNotFoundError(f"Không tìm thấy used_cars.csv trong: {[str(p) for p in candidate_paths]}")
processed_json_path = processed_dir / "car_data.json"
processed_csv_path = processed_dir / "processed_car_data.csv"

print(f"Làm sạch dữ liệu từ {used_cars_path}...")

# Đọc CSV với error handling
try:
    raw_df = pd.read_csv(used_cars_path, encoding='utf-8-sig')
except Exception as e:
    print(f"Lỗi đọc CSV: {e}, thử bỏ qua dòng lỗi...")
    raw_df = pd.read_csv(used_cars_path, encoding='utf-8-sig', on_bad_lines='skip')

raw_cars = raw_df.to_dict(orient='records')
print(f"Đọc {len(raw_cars)} xe từ CSV")

# Danh sách hãng xe
car_brands = [
    "Toyota", "Honda", "Ford", "Mazda", "Kia", "Hyundai", "Chevrolet", 
    "BMW", "Mercedes-Benz", "Mercedes", "Benz", "Audi", "Lexus", "Nissan", 
    "Mitsubishi", "Suzuki", "Volkswagen", "Porsche", "Subaru", "Isuzu", 
    "Volvo", "Peugeot", "Mini", "Land Rover", "Range Rover", "Jaguar", 
    "Renault", "Jeep", "Bentley", "Rolls-Royce", "Ferrari", "Lamborghini", 
    "Maserati", "Tesla", "BYD", "MG", "Vinfast"
]

def extract_car_brand(car_name):
    """Trích xuất hãng xe."""
    if not car_name:
        return None
    
    car_name = re.sub(r'^(Xe|xe)\s+', '', car_name, flags=re.IGNORECASE)
    
    # Tìm hãng trong danh sách
    for brand in car_brands:
        if re.search(r'\b' + re.escape(brand) + r'\b', car_name, re.IGNORECASE):
            return brand
    
    # Lấy từ đầu tiên hợp lệ
    words = car_name.split()
    for word in words[:2]:  # Chỉ kiểm tra 2 từ đầu
        if word and word.lower() not in ["bán", "cần", "đang"]:
            return word
    
    return None

def get_doors_by_body_type(body_type):
    """Xác định số cửa theo kiểu dáng."""
    body_type = body_type.lower() if body_type else ""
    door_map = {
        "sedan": "4", "coupe": "2", "suv": "5", "crossover": "5",
        "bán tải": "4", "pickup": "4", "hatchback": "5", 
        "van": "5", "minivan": "5", "convertible": "2", "cabriolet": "2"
    }
    return next((doors for key, doors in door_map.items() if key in body_type), "4")

# Xử lý dữ liệu
processed_cars = []
for car in raw_cars:
    processed_car = {
        "ten_xe": car.get("Ten_xe", ""),
        "hang_xe": extract_car_brand(car.get("Ten_xe", "")),
        "gia_ban": car.get("Gia_ban_text", ""),
        "gia_ban_number": car.get("Gia_ban", ""),
        "url": car.get("Link", ""),
        "nam_sx": car.get("Nam_san_xuat", ""),
        "xuat_xu": car.get("Xuat_xu", ""),
        "hop_so": car.get("Hop_so", ""),
        "kieu_dang": car.get("Kieu_dang", ""),
        "dong_co": car.get("Dong_co", ""),
        "dan_dong": car.get("Dan_dong", ""),
        "so_km_da_di": car.get("So_km", ""),
        "so_cho_ngoi": car.get("So_cho_ngoi", ""),
        "so_cua": "",
        "mau_ngoai_that": car.get("Mau_ngoai_that", ""),
        "mau_noi_that": car.get("Mau_noi_that", ""),
        "tinh_trang": "Xe đã dùng",
    }
        
    # Làm sạch số chỗ ngồi và km
    if processed_car["so_cho_ngoi"]:
        processed_car["so_cho_ngoi"] = re.sub(r'\s*chỗ.*', '', processed_car["so_cho_ngoi"]).strip()
    
    if processed_car["so_km_da_di"]:
        processed_car["so_km_da_di"] = re.sub(r'\s*Km.*', '', processed_car["so_km_da_di"]).strip().replace(',', '')
    
    # Xác định số cửa
    processed_car["so_cua"] = get_doors_by_body_type(processed_car["kieu_dang"])
    
    processed_cars.append(processed_car)

print(f"Xử lý {len(processed_cars)} xe")

# Lưu Vietnamese version
with open(processed_json_path, 'w', encoding='utf-8') as f:
    json.dump(processed_cars, f, ensure_ascii=False, indent=2)

pd.DataFrame(processed_cars).to_csv(processed_csv_path, index=False, encoding='utf-8-sig')

# Tạo English version
english_mapping = {
    "ten_xe": "name", "hang_xe": "brand", "gia_ban": "price", "url": "url",
    "nam_sx": "year", "xuat_xu": "origin", "hop_so": "transmission", 
    "kieu_dang": "body_type", "dong_co": "engine", "dan_dong": "drive",
    "so_km_da_di": "mileage_km", "so_cho_ngoi": "seats", "so_cua": "doors",
    "mau_ngoai_that": "exterior_color", "mau_noi_that": "interior_color", 
    "tinh_trang": "condition"
}

english_cars = [
    {english_mapping[k]: v for k, v in car.items() if k in english_mapping}
    for car in processed_cars
]

# Lưu English version
english_json_path = processed_dir / "car_data_en.json" 
english_csv_path = processed_dir / "car_data_en.csv"

with open(english_json_path, 'w', encoding='utf-8') as f:
    json.dump(english_cars, f, ensure_ascii=False, indent=2)

pd.DataFrame(english_cars).to_csv(english_csv_path, index=False, encoding='utf-8-sig')

print(f"Lưu thành công:")
print(f"- VN: {processed_json_path}")
print(f"- EN: {english_json_path}")
print("Hoàn tất! Chạy enhance_car_data.py tiếp theo.")