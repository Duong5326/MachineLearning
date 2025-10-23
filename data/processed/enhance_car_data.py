import json
import pandas as pd
import re
import os
from datetime import datetime

def load_data(file_path):
    """Đọc dữ liệu từ file JSON."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
def convert_price_to_number(price_text, price_number=None):
    """
    Chuyển đổi giá từ dạng văn bản (vd: 1 Tỷ 599 Triệu) sang số (đơn vị: triệu).
    Nếu price_number đã có (từ trường Gia_ban trong used_cars.csv), sử dụng giá trị đó.
    """
    # Nếu đã có giá trị số, chuyển đổi từ VND sang triệu đồng
    if price_number and str(price_number).strip():
        try:
            return int(float(price_number) / 1000000)  # Chuyển từ VND sang triệu
        except (ValueError, TypeError):
            pass
    
    # Nếu không có giá trị số hoặc không thể chuyển đổi, dùng cách cũ
    price_text = str(price_text).strip()
    ty_pattern = r'(\d+)\s*Tỷ'
    trieu_pattern = r'(\d+)\s*Triệu'
    
    price_in_million = 0
    
    # Tìm số tỷ
    ty_match = re.search(ty_pattern, price_text)
    if ty_match:
        price_in_million += int(ty_match.group(1)) * 1000
    
    # Tìm số triệu
    trieu_match = re.search(trieu_pattern, price_text)
    if trieu_match:
        price_in_million += int(trieu_match.group(1))
    
    return price_in_million

def extract_engine_info(engine_text):
    """Tách thông tin động cơ thành loại nhiên liệu và dung tích."""
    if pd.isna(engine_text) or not engine_text:
        return {"nhien_lieu": "", "dung_tich": ""}
    
    engine_text = str(engine_text).strip()
    
    # Xác định loại nhiên liệu
    nhien_lieu = ""
    if "Xăng" in engine_text:
        nhien_lieu = "Xăng"
    elif "Dầu" in engine_text:
        nhien_lieu = "Dầu"
    elif "Điện" in engine_text:
        nhien_lieu = "Điện"
    elif "Hybrid" in engine_text:
        nhien_lieu = "Hybrid"
    
    # Trích xuất dung tích động cơ
    dung_tich = ""
    capacity_match = re.search(r'(\d+\.\d+|\d+)\s*L', engine_text)
    if capacity_match:
        dung_tich = capacity_match.group(1)
    
    return {"nhien_lieu": nhien_lieu, "dung_tich": dung_tich}

def extract_transmission_info(ten_xe, hop_so=None):
    """Trích xuất thông tin hộp số từ tên xe và trường hộp số nếu có."""
    # Nếu đã có thông tin hộp số, sử dụng nó
    if hop_so and isinstance(hop_so, str) and hop_so.strip():
        hop_so_lower = hop_so.lower()
        if "tự động" in hop_so_lower:
            return "Số tự động"
        elif "sàn" in hop_so_lower:
            return "Số sàn"
        elif "cvt" in hop_so_lower:
            return "Số vô cấp CVT"
        elif "dct" in hop_so_lower:
            return "Số ly hợp kép DCT"
        else:
            return hop_so
    
    # Nếu không có, trích xuất từ tên xe
    ten_xe = str(ten_xe).upper()
    
    if "AT" in ten_xe:
        return "Số tự động"
    elif "MT" in ten_xe:
        return "Số sàn"
    elif "CVT" in ten_xe:
        return "Số vô cấp CVT"
    elif "DCT" in ten_xe:
        return "Số ly hợp kép DCT"
    else:
        return ""

def extract_drive_system(ten_xe, dan_dong=None):
    """Trích xuất thông tin hệ dẫn động từ tên xe và trường dẫn động nếu có."""
    # Nếu đã có thông tin dẫn động, sử dụng nó
    if dan_dong and isinstance(dan_dong, str) and dan_dong.strip():
        dan_dong_lower = dan_dong.lower()
        if "4wd" in dan_dong_lower or "awd" in dan_dong_lower or "4 bánh" in dan_dong_lower:
            return dan_dong
        elif "fwd" in dan_dong_lower or "cầu trước" in dan_dong_lower:
            return dan_dong
        elif "rwd" in dan_dong_lower or "rfd" in dan_dong_lower or "cầu sau" in dan_dong_lower:
            return dan_dong
        else:
            return dan_dong
    
    # Nếu không có, trích xuất từ tên xe
    ten_xe = str(ten_xe).upper()
    
    if "4X4" in ten_xe or "4WD" in ten_xe or "AWD" in ten_xe:
        return "4 bánh"
    elif "4X2" in ten_xe or "2WD" in ten_xe:
        return "2 bánh"
    elif "4MATIC" in ten_xe:
        return "4 bánh (4Matic)"
    else:
        return ""

def classify_car_by_price(price):
    """Phân loại xe theo khoảng giá."""
    if price < 500:
        return "Phổ thông"
    elif price < 1000:
        return "Trung cấp"
    elif price < 2000:
        return "Cao cấp"
    else:
        return "Sang trọng"

def classify_car_by_segment(kieu_dang, hang_xe, gia):
    """Phân loại xe theo phân khúc."""
    if kieu_dang == "Sedan":
        if gia < 500:
            return "Sedan hạng B"
        elif gia < 1000:
            return "Sedan hạng C"
        else:
            return "Sedan hạng D+"
    elif kieu_dang == "SUV":
        if gia < 600:
            return "SUV cỡ nhỏ"
        elif gia < 1200:
            return "SUV cỡ trung"
        else:
            return "SUV cỡ lớn"
    elif kieu_dang == "Crossover":
        if gia < 600:
            return "Crossover cỡ nhỏ"
        else:
            return "Crossover cỡ trung"
    else:
        return kieu_dang

def calculate_car_age(year, current_year=2025):
    """Tính tuổi xe dựa vào năm sản xuất."""
    try:
        year = int(year)
        age = current_year - year
        return age
    except (ValueError, TypeError):
        return None

def enhance_car_data(cars):
    """Nâng cao dữ liệu xe."""
    current_year = datetime.now().year
    enhanced_cars = []
    
    for car in cars:
        # Tạo bản sao của dữ liệu xe
        enhanced_car = car.copy()
        
        # Chuyển đổi giá bán sang số (sử dụng cả gia_ban_text và gia_ban_number nếu có)
        if "gia_ban" in car:
            price_in_million = convert_price_to_number(car["gia_ban"], car.get("gia_ban_number", None))
            enhanced_car["gia_trieu"] = price_in_million
        
        # Tách thông tin động cơ
        if "dong_co" in car:
            engine_info = extract_engine_info(car["dong_co"])
            enhanced_car["nhien_lieu"] = engine_info["nhien_lieu"]
            enhanced_car["dung_tich"] = engine_info["dung_tich"]
        
        # Trích xuất thông tin hộp số
        if "ten_xe" in car:
            if not enhanced_car.get("hop_so") and "hop_so" in car:
                enhanced_car["hop_so"] = car["hop_so"]
            else:
                enhanced_car["hop_so"] = extract_transmission_info(car["ten_xe"], car.get("hop_so", ""))
        
        # Trích xuất thông tin dẫn động
        if "ten_xe" in car:
            if not enhanced_car.get("dan_dong") and "dan_dong" in car:
                enhanced_car["dan_dong"] = car["dan_dong"]
            else:
                enhanced_car["dan_dong"] = extract_drive_system(car["ten_xe"], car.get("dan_dong", ""))
        
        # Tính tuổi xe
        if "nam_sx" in car:
            enhanced_car["tuoi_xe"] = calculate_car_age(car["nam_sx"], current_year)
        
        # Phân loại xe theo giá
        if "gia_trieu" in enhanced_car:
            enhanced_car["phan_khuc_gia"] = classify_car_by_price(enhanced_car["gia_trieu"])
        
        # Phân loại xe theo phân khúc
        if "kieu_dang" in car and "hang_xe" in car and "gia_trieu" in enhanced_car:
            enhanced_car["phan_khuc"] = classify_car_by_segment(
                car["kieu_dang"], car["hang_xe"], enhanced_car["gia_trieu"]
            )
        
        enhanced_cars.append(enhanced_car)
    
    return enhanced_cars

def map_column_names(cars):
    """Cập nhật tên cột để phù hợp với định dạng mới từ bonbanh_collector.py."""
    mapped_cars = []
    for car in cars:
        mapped_car = {}
        # Bản đồ ánh xạ tên cột mới sang tên cột cũ
        field_mapping = {
            "name": "ten_xe",
            "brand": "hang_xe",
            "price": "gia_ban",
            "url": "url",
            "year": "nam_sx",
            "body_type": "kieu_dang",
            "condition": "tinh_trang",
            "origin": "xuat_xu",
            "transmission": "hop_so",
            "drive": "dan_dong",
            "seats": "so_cho_ngoi",
            "doors": "so_cua",
            "mileage_km": "so_km_da_di",
            "exterior_color": "mau_ngoai_that",
            "interior_color": "mau_noi_that",
            "fuel": "nhien_lieu"
        }
        
        for new_field, old_field in field_mapping.items():
            if new_field in car:
                mapped_car[old_field] = car[new_field]
        
        mapped_cars.append(mapped_car)
    
    return mapped_cars

def main():
    # Đường dẫn file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Xác định đường dẫn các file
    input_json_en = os.path.join(script_dir, "car_data_en.json")
    input_json = os.path.join(script_dir, "car_data.json")
    output_json = os.path.join(script_dir, "enhanced_car_data.json")
    output_csv = os.path.join(script_dir, "enhanced_car_data.csv")
    
    # Chạy script làm sạch dữ liệu trước để tạo ra car_data.json nếu chưa có
    if not os.path.exists(input_json) and not os.path.exists(input_json_en):
        print("Không tìm thấy file dữ liệu đã làm sạch. Chạy script clean_raw_to_processed.py...")
        
        # Sử dụng importlib.util để gọi script khác
        import importlib.util
        import sys
        
        clean_script_path = os.path.join(script_dir, "clean_raw_to_processed.py")
        spec = importlib.util.spec_from_file_location("clean_module", clean_script_path)
        clean_module = importlib.util.module_from_spec(spec)
        sys.modules["clean_module"] = clean_module
        spec.loader.exec_module(clean_module)
        
        print("Hoàn thành làm sạch dữ liệu.")
    
    # Ưu tiên sử dụng file tiếng Anh nếu có
    if os.path.exists(input_json_en):
        print(f"Đọc dữ liệu từ {input_json_en}...")
        cars = load_data(input_json_en)
        print(f"Đã đọc {len(cars)} xe từ file tiếng Anh.")
        # File tiếng Anh đã có các trường với tên tiếng Anh rồi, không cần ánh xạ
    else:
        # Đọc dữ liệu
        print(f"Đọc dữ liệu từ {input_json}...")
        cars = load_data(input_json)
        print(f"Đã đọc {len(cars)} xe.")
        
        # Ánh xạ tên cột mới sang tên cột cũ
        cars = map_column_names(cars)
    
    # Làm sạch và nâng cao dữ liệu
    print("Làm sạch và nâng cao dữ liệu...")
    enhanced_cars = enhance_car_data(cars)
    
    # Lưu dữ liệu đã cải thiện
    print(f"Lưu dữ liệu đã cải thiện vào {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(enhanced_cars, f, ensure_ascii=False, indent=2)
    
    # Tạo DataFrame và lưu ra CSV
    df = pd.DataFrame(enhanced_cars)
    
    # Sắp xếp các cột theo thứ tự hợp lý
    columns_order = [
        "ten_xe", "hang_xe", "phan_khuc", "gia_ban", "gia_trieu", "phan_khuc_gia",
        "nam_sx", "tuoi_xe", "xuat_xu", "kieu_dang", 
        "nhien_lieu", "dung_tich", "dong_co", "hop_so", "dan_dong",
        "mau_ngoai_that", "mau_noi_that", "tinh_trang", "url"
    ]
    
    # Chỉ giữ lại các cột hiện có
    columns_order = [col for col in columns_order if col in df.columns]
    
    # Thêm bất kỳ cột nào còn thiếu trong danh sách trên
    for col in df.columns:
        if col not in columns_order:
            columns_order.append(col)
    
    df = df[columns_order]
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Đã lưu {len(df)} xe vào {output_csv}")
    
    # Hiển thị mẫu dữ liệu
    print("\n=== Mẫu dữ liệu đã cải thiện ===")
    print(df.head(3))
    
    # Thống kê cơ bản
    print("\n=== Thống kê cơ bản ===")
    print(f"Số lượng xe: {len(df)}")
    
    # Kiểm tra tên cột phù hợp (hang_xe hoặc brand)
    if 'hang_xe' in df.columns:
        brand_col = 'hang_xe'
    elif 'brand' in df.columns:
        brand_col = 'brand'
    else:
        brand_col = None
    
    if brand_col:
        print(f"Phân bố theo hãng xe:\n{df[brand_col].value_counts()}")
    
    if 'phan_khuc_gia' in df.columns:
        print(f"Phân bố theo phân khúc giá:\n{df['phan_khuc_gia'].value_counts()}")
    
    if 'nhien_lieu' in df.columns:
        print(f"Phân bố theo loại nhiên liệu:\n{df['nhien_lieu'].value_counts()}")
    elif 'fuel' in df.columns:
        print(f"Phân bố theo loại nhiên liệu:\n{df['fuel'].value_counts()}")
    
    # Giá trung bình theo hãng xe
    print("\n=== Giá trung bình theo hãng xe ===")
    price_col = 'gia_trieu' if 'gia_trieu' in df.columns else 'price_million'
    if brand_col and price_col in df.columns:
        print(df.groupby(brand_col)[price_col].mean().sort_values(ascending=False))

if __name__ == "__main__":
    main()