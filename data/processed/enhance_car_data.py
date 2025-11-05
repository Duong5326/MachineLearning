import json
import pandas as pd
import re
import os
from datetime import datetime

def load_data(file_path):
    """Đọc dữ liệu từ file JSON."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def clean_car_name(name):
    """Làm sạch tên xe, loại bỏ đuôi kỹ thuật."""
    return re.sub(r'\s+\.[0-9]+[A-Z]+', '', name.strip()) if name else name
def convert_price_to_number(price_text, price_number=None):
    """Chuyển đổi giá sang triệu đồng."""
    if price_number and str(price_number).strip():
        try:
            return int(float(price_number) / 1000000)
        except (ValueError, TypeError):
            pass
    
    price_text = str(price_text).strip()
    price_in_million = 0
    
    ty_match = re.search(r'(\d+)\s*Tỷ', price_text)
    if ty_match:
        price_in_million += int(ty_match.group(1)) * 1000
    
    trieu_match = re.search(r'(\d+)\s*Triệu', price_text)
    if trieu_match:
        price_in_million += int(trieu_match.group(1))
    
    return price_in_million

def extract_engine_info(engine_text):
    """Tách thông tin động cơ."""
    if pd.isna(engine_text) or not engine_text:
        return {"nhien_lieu": "", "dung_tich": ""}
    
    engine_text = str(engine_text).strip()
    
    # Loại nhiên liệu
    fuel_types = {"Xăng": "Xăng", "Dầu": "Dầu", "Điện": "Điện", "Hybrid": "Hybrid"}
    nhien_lieu = next((v for k, v in fuel_types.items() if k in engine_text), "")
    
    # Dung tích
    capacity_match = re.search(r'(\d+\.\d+|\d+)\s*L', engine_text)
    dung_tich = capacity_match.group(1) if capacity_match else ""
    
    return {"nhien_lieu": nhien_lieu, "dung_tich": dung_tich}

def extract_transmission_info(ten_xe, hop_so=None):
    """Trích xuất thông tin hộp số."""
    if hop_so and isinstance(hop_so, str) and hop_so.strip():
        hop_so_lower = hop_so.lower()
        return "Số tự động" if any(x in hop_so_lower for x in ["tự động", "cvt", "dct"]) else \
               "Số sàn" if "sàn" in hop_so_lower else "Số tự động"
    
    ten_xe_upper = str(ten_xe).upper()
    return "Số tự động" if any(x in ten_xe_upper for x in ["AT", "CVT", "DCT"]) else \
           "Số sàn" if "MT" in ten_xe_upper else ""

def extract_drive_system(ten_xe, dan_dong=None):
    """Trích xuất hệ dẫn động."""
    if dan_dong and isinstance(dan_dong, str) and dan_dong.strip():
        return dan_dong
    
    ten_xe_upper = str(ten_xe).upper()
    if any(x in ten_xe_upper for x in ["4X4", "4WD", "AWD"]):
        return "4 bánh"
    elif any(x in ten_xe_upper for x in ["4X2", "2WD"]):
        return "2 bánh"
    elif "4MATIC" in ten_xe_upper:
        return "4 bánh (4Matic)"
    return ""

def calculate_car_age(year, current_year=2025):
    """Tính tuổi xe."""
    try:
        return current_year - int(year)
    except (ValueError, TypeError):
        return 0

def enhance_car_data(cars):
    """Nâng cao dữ liệu xe."""
    enhanced_cars = []
    
    for car in cars:
        enhanced_car = car.copy()
        
        # Làm sạch tên xe
        if car.get("name"):
            enhanced_car["name"] = clean_car_name(car["name"])
        
        # Chuyển đổi giá
        if car.get("price"):
            enhanced_car["price_million"] = convert_price_to_number(car["price"]) or 0
        
        # Tách thông tin động cơ
        if car.get("engine"):
            engine_info = extract_engine_info(car["engine"])
            enhanced_car["fuel_type"] = engine_info["nhien_lieu"] or "Unknown"
            enhanced_car["engine_capacity"] = engine_info["dung_tich"] or "0"
        
        # Tính tuổi xe
        if car.get("year"):
            enhanced_car["car_age"] = calculate_car_age(car["year"])
        
        # Chuẩn hóa mileage
        if "mileage_km" in car:
            try:
                enhanced_car["mileage_km"] = int(float(car["mileage_km"]))
            except:
                enhanced_car["mileage_km"] = 0
        
        enhanced_cars.append(enhanced_car)
    
    return enhanced_cars

def map_column_names(cars):
    """Ánh xạ tên cột."""
    field_mapping = {
        "name": "ten_xe", "brand": "hang_xe", "price": "gia_ban", "url": "url",
        "year": "nam_sx", "body_type": "kieu_dang", "condition": "tinh_trang",
        "origin": "xuat_xu", "transmission": "hop_so", "drive": "dan_dong",
        "seats": "so_cho_ngoi", "doors": "so_cua", "mileage_km": "so_km_da_di",
        "exterior_color": "mau_ngoai_that", "interior_color": "mau_noi_that", "fuel": "nhien_lieu"
    }
    
    return [{old: car.get(new, "") for new, old in field_mapping.items()} for car in cars]

def sync_field_order(enhanced_cars):
    """Đồng bộ thứ tự trường."""
    fields = [
        "url", "name", "brand", "price", "price_million",
        "year", "car_age", "origin", "transmission", "body_type",
        "engine", "fuel_type", "engine_capacity", "drive", 
        "mileage_km", "seats", "doors", "exterior_color", 
        "interior_color", "condition"
    ]
    
    numeric_fields = {"price_million", "car_age", "mileage_km", "seats", "doors"}
    
    return [
        {field: car.get(field, 0 if field in numeric_fields else "") for field in fields}
        for car in enhanced_cars
    ]

def compare_file_structures(csv_file, json_file):
    """Kiểm tra đồng bộ CSV/JSON."""
    try:
        df = pd.read_csv(csv_file)
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        if not (json_data and list(json_data[0].keys()) == df.columns.tolist() and len(json_data) == len(df)):
            print(" CSV/JSON chưa đồng bộ")
    except:
        print(" Lỗi kiểm tra file")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_json_en = os.path.join(script_dir, "car_data_en.json")
    input_json = os.path.join(script_dir, "car_data.json")
    output_json = os.path.join(script_dir, "enhanced_car_data.json")
    output_csv = os.path.join(script_dir, "enhanced_car_data.csv")
    
    # Auto-run clean script if needed
    if not os.path.exists(input_json) and not os.path.exists(input_json_en):
        print("Chạy script làm sạch dữ liệu...")
        import importlib.util
        import sys
        
        clean_script_path = os.path.join(script_dir, "clean_raw_to_processed.py")
        spec = importlib.util.spec_from_file_location("clean_module", clean_script_path)
        clean_module = importlib.util.module_from_spec(spec)
        sys.modules["clean_module"] = clean_module
        spec.loader.exec_module(clean_module)
        print("Hoàn thành làm sạch.")
    
    # Load and process data
    cars = load_data(input_json_en if os.path.exists(input_json_en) else input_json)
    if not os.path.exists(input_json_en):
        cars = map_column_names(cars)
    
    print(f"Xử lý {len(cars)} xe...")
    
    enhanced_cars = enhance_car_data(cars)
    standardized_cars = sync_field_order(enhanced_cars)
    
    # Save files
    df = pd.DataFrame(standardized_cars)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(standardized_cars, f, ensure_ascii=False, indent=2)
    
    compare_file_structures(output_csv, output_json)
    print(f" Hoàn thành: {len(df)} xe, {len(df.columns)} trường")

if __name__ == "__main__":
    main()