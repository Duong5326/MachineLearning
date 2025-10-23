"""
Script để làm sạch dữ liệu xe từ thư mục raw và lưu vào thư mục processed
Script này sẽ:
1. Đọc dữ liệu thô từ data/raw/car_data.json
2. Loại bỏ các trường trùng lặp và tiêu chuẩn hóa dữ liệu
3. Lưu dữ liệu đã làm sạch vào data/processed/car_data.json và CSV
"""

import json
import os
import pandas as pd
from pathlib import Path
import re

# Thiết lập đường dẫn
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir.parent.parent  # MachineLearning folder
processed_dir = project_root / "data" / "processed"

# Đường dẫn file - sử dụng đường dẫn tuyệt đối đến file used_cars.csv trong thư mục data/raw/data/raw
used_cars_path = Path(r"d:\New folder\MachineLearning\data\raw\data\raw\used_cars.csv")
processed_json_path = processed_dir / "car_data.json"
processed_csv_path = processed_dir / "processed_car_data.csv"

print(f"Bắt đầu làm sạch dữ liệu xe từ {used_cars_path}...")

# Đọc dữ liệu từ CSV với error_bad_lines=False (skiprows với engine='python')
try:
    # Thử đọc với cách thông thường trước
    raw_df = pd.read_csv(used_cars_path, encoding='utf-8-sig')
except Exception as e:
    print(f"Gặp lỗi khi đọc CSV: {e}")
    print("Thử đọc với tham số on_bad_lines='skip'...")
    # Nếu gặp lỗi, thử đọc bỏ qua các dòng có vấn đề
    raw_df = pd.read_csv(used_cars_path, encoding='utf-8-sig', on_bad_lines='skip')

print(f"Đã đọc {len(raw_df)} xe từ CSV")

# Chuyển đổi từ DataFrame sang danh sách dict
raw_cars = raw_df.to_dict(orient='records')

print(f"Đã đọc {len(raw_cars)} xe từ dữ liệu thô")

# Map từ trường trong used_cars.csv sang trường chuẩn
field_mapping = {
    "Ten_xe": "ten_xe",
    "Nam_san_xuat": "nam_sx",
    "So_km": "so_km_da_di",
    "Xuat_xu": "xuat_xu",
    "Kieu_dang": "kieu_dang",
    "Hop_so": "hop_so",
    "Dong_co": "dong_co",
    "Mau_ngoai_that": "mau_ngoai_that",
    "Mau_noi_that": "mau_noi_that",
    "So_cho_ngoi": "so_cho_ngoi",
    "Dan_dong": "dan_dong",
    "Gia_ban_text": "gia_ban",
    "Gia_ban": "gia_ban_number",
    "Link": "url"
}

# Danh sách các hãng xe phổ biến để đối chiếu
car_brands = [
    "Toyota", "Honda", "Ford", "Mazda", "Kia", "Hyundai", "Chevrolet", 
    "BMW", "Mercedes-Benz", "Mercedes", "Benz", "Audi", "Lexus", "Nissan", 
    "Mitsubishi", "Suzuki", "Volkswagen", "Porsche", "Subaru", "Isuzu", 
    "Volvo", "Peugeot", "Mini", "Land Rover", "Range Rover", "Jaguar", 
    "Renault", "Jeep", "Bentley", "Rolls-Royce", "Ferrari", "Lamborghini", 
    "Maserati", "Tesla", "BYD", "MG", "Vinfast"
]

def extract_car_brand(car_name):
    """Trích xuất hãng xe từ tên xe."""
    if not car_name:
        return None
    
    # Loại bỏ từ "Xe" hoặc "xe" ở đầu nếu có
    car_name = re.sub(r'^(Xe|xe)\s+', '', car_name, flags=re.IGNORECASE)
    
    # Thử tìm hãng xe trong danh sách đã biết
    for brand in car_brands:
        pattern = r'\b' + re.escape(brand) + r'\b'
        if re.search(pattern, car_name, re.IGNORECASE):
            return brand
    
    # Nếu không tìm thấy, lấy từ đầu tiên
    first_word = car_name.split()[0].strip()
    if first_word and first_word.lower() not in ["bán", "cần", "đang"]:
        return first_word
    
    # Nếu từ đầu tiên không phù hợp, thử lấy từ thứ hai
    words = car_name.split()
    if len(words) > 1 and words[1].lower() not in ["bán", "cần", "đang"]:
        return words[1]
    
    return None

# Làm sạch dữ liệu
processed_cars = []
for car in raw_cars:
    # Tạo đối tượng xe mới với các trường được ánh xạ từ used_cars.csv
    processed_car = {
        "ten_xe": car.get("Ten_xe", ""),
        "hang_xe": extract_car_brand(car.get("Ten_xe", "")),  # Trích xuất hãng xe từ tên
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
        "so_cua": "",  # Không có trong used_cars.csv
        "mau_ngoai_that": car.get("Mau_ngoai_that", ""),
        "mau_noi_that": car.get("Mau_noi_that", ""),
        "tinh_trang": "Xe đã dùng",  # Giá trị mặc định
    }
        
    # Loại bỏ ký tự chỗ từ số chỗ ngồi (vd: "5 chỗ" -> "5")
    if processed_car["so_cho_ngoi"]:
        processed_car["so_cho_ngoi"] = re.sub(r'\s*chỗ.*', '', processed_car["so_cho_ngoi"]).strip()
    
    # Loại bỏ "Km" từ số km đã đi
    if processed_car["so_km_da_di"]:
        processed_car["so_km_da_di"] = re.sub(r'\s*Km.*', '', processed_car["so_km_da_di"]).strip()
        # Loại bỏ dấu phẩy trong số
        processed_car["so_km_da_di"] = processed_car["so_km_da_di"].replace(',', '')
    
    # Điền thông tin số cửa dựa trên kiểu dáng xe
    if not processed_car["so_cua"]:
        kieu_dang = processed_car["kieu_dang"].lower() if processed_car["kieu_dang"] else ""
        if "sedan" in kieu_dang:
            processed_car["so_cua"] = "4"
        elif "coupe" in kieu_dang:
            processed_car["so_cua"] = "2"
        elif "suv" in kieu_dang or "crossover" in kieu_dang:
            processed_car["so_cua"] = "5"
        elif "bán tải" in kieu_dang or "pickup" in kieu_dang:
            processed_car["so_cua"] = "4"
        elif "hatchback" in kieu_dang:
            processed_car["so_cua"] = "5"
        elif "van" in kieu_dang or "minivan" in kieu_dang:
            processed_car["so_cua"] = "5"
        elif "convertible" in kieu_dang or "cabriolet" in kieu_dang:
            processed_car["so_cua"] = "2"
        else:
            # Mặc định nếu không nhận dạng được kiểu dáng
            processed_car["so_cua"] = "4"
    
    processed_cars.append(processed_car)

print(f"Đã xử lý và làm sạch {len(processed_cars)} xe")

# Lưu dữ liệu đã làm sạch vào thư mục processed
with open(processed_json_path, 'w', encoding='utf-8') as file:
    json.dump(processed_cars, file, ensure_ascii=False, indent=2)

# Tạo DataFrame và lưu ra CSV
df = pd.DataFrame(processed_cars)
df.to_csv(processed_csv_path, index=False, encoding='utf-8-sig')

print(f"Đã lưu dữ liệu đã làm sạch vào:")
print(f"- JSON: {processed_json_path}")
print(f"- CSV: {processed_csv_path}")

# Tạo file English version cho data science
english_cars = []
for car in processed_cars:
    english_car = {
        "name": car.get("ten_xe", ""),
        "brand": car.get("hang_xe", ""),
        "price": car.get("gia_ban", ""),
        "url": car.get("url", ""),
        "year": car.get("nam_sx", ""),
        "origin": car.get("xuat_xu", ""),
        "transmission": car.get("hop_so", ""),
        "body_type": car.get("kieu_dang", ""),
        "engine": car.get("dong_co", ""),
        "drive": car.get("dan_dong", ""),
        "mileage_km": car.get("so_km_da_di", ""),
        "seats": car.get("so_cho_ngoi", ""),
        "doors": car.get("so_cua", ""),
        "exterior_color": car.get("mau_ngoai_that", ""),
        "interior_color": car.get("mau_noi_that", ""),
        "condition": car.get("tinh_trang", ""),
    }
    english_cars.append(english_car)

# Lưu phiên bản tiếng Anh
english_json_path = processed_dir / "car_data_en.json"
english_csv_path = processed_dir / "car_data_en.csv"

with open(english_json_path, 'w', encoding='utf-8') as file:
    json.dump(english_cars, file, ensure_ascii=False, indent=2)

df_en = pd.DataFrame(english_cars)
df_en.to_csv(english_csv_path, index=False, encoding='utf-8-sig')

print(f"Đã lưu phiên bản tiếng Anh cho phân tích dữ liệu:")
print(f"- JSON: {english_json_path}")
print(f"- CSV: {english_csv_path}")

print("\nHoàn tất quá trình làm sạch dữ liệu!")
print("Bạn có thể tiếp tục chạy enhance_car_data.py để nâng cao dữ liệu.")