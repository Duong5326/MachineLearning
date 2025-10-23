"""
Ứng dụng Flask để dự đoán giá xe dựa trên mô hình học máy đã huấn luyện 
bằng wed ():):())
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Sử dụng backend non-interactive
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

app = Flask(__name__)

# Đường dẫn đến mô hình
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "src", "models", "Lasso_model.pkl")

# Đường dẫn đến dữ liệu
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "data", "processed", "enhanced_car_data.json")
# Nạp dữ liệu
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_paths = [
        os.path.join(base_dir, 'data', 'processed', 'enhanced_car_data.csv'),
        os.path.join(base_dir, 'data', 'processed', 'processed_car_data.csv'),
        os.path.join(base_dir, 'data', 'processed', 'car_data_en.csv'),
        os.path.join(base_dir, 'data', 'raw', 'used_cars.csv')
    ]

    for path in csv_paths:
        if os.path.isfile(path):
            try:
                df = pd.read_csv(path, encoding='utf-8')
                print(f" Đã nạp dữ liệu từ: {path}, số dòng: {len(df)}")
                return df  # Chỉ lấy file đầu tiên hợp lệ
            except Exception as e:
                print(f" Lỗi khi đọc {path}: {e}")

    #Nếu không có CSV nào, thử JSON
    json_paths = [
        os.path.join(base_dir, 'data', 'processed', 'enhanced_car_data.json'),
        os.path.join(base_dir, 'data', 'processed', 'car_data_en.json'),
        os.path.join(base_dir, 'data', 'processed', 'car_data.json')
    ]
    for path in json_paths:
        if os.path.isfile(path):
            try:
                df = pd.read_json(path)
                print(f" Đã nạp dữ liệu từ: {path}")
                return df
            except Exception as e:
                print(f" Lỗi khi đọc {path}: {e}")

    print(" Không tìm thấy dữ liệu hợp lệ.")
    return pd.DataFrame()



# Nạp mô hình
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        print(f"Lỗi khi nạp mô hình: {str(e)}")
        return None

# Hàm chuyển đổi trường 'price' dạng text sang cột 'gia_trieu' số (triệu VND)
def parse_price_text_to_million(price_str):
    if pd.isnull(price_str):
        return None
    s = str(price_str).lower().replace(' ', '')
    if 'tỷ' in s:
        parts = s.split('tỷ')
        ty = float(parts[0]) if parts[0] else 0
        trieu = 0
        if len(parts) > 1 and 'triệu' in parts[1]:
            trieu_part = parts[1].replace('triệu','')
            trieu = float(trieu_part) if trieu_part else 0
        return ty * 1000 + trieu
    elif 'triệu' in s:
        return float(s.replace('triệu',''))
    elif 'trieu' in s:
        return float(s.replace('trieu',''))
    else:
        try:
            return float(s)
        except:
            return None

# Nạp dữ liệu

@app.route('/')
def home():
    """Trang chủ với form nhập liệu"""
    # Chuẩn bị dữ liệu cho các trường select
    brands = ["Toyota", "Honda", "Mazda", "Ford", "Kia", "Hyundai", 
              "Mercedes-Benz", "BMW", "Lexus", "Audi", "Vinfast", "Khác"]
    
    body_types = ["Sedan", "SUV", "Hatchback", "Crossover", "MPV", "Pickup", "Van"]
    
    fuel_types = ["Gasoline", "Diesel", "Electric", "Hybrid"]
    
    # Truyền dữ liệu đến template
    return render_template('index.html', 
                          brands=brands, 
                          body_types=body_types,
                          fuel_types=fuel_types,
                          current_year=datetime.now().year)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Nhận dữ liệu từ form và dự đoán giá"""
    if request.method == 'POST':
        # Bắt buộc nhập số km và loại số
        required_fields = ['brand', 'year', 'body_type', 'engine_capacity', 'fuel_type', 'is_imported', 'transmission', 'mileage']
        missing = [f for f in required_fields if not request.form.get(f, '').strip()]
        if missing:
            return render_template('error.html', error=f"Vui lòng nhập đầy đủ các trường: {', '.join(missing)}")
        try:
            brand = request.form['brand']
            year = int(request.form['year'])
            body_type = request.form['body_type']
            engine_capacity = float(request.form['engine_capacity'])
            fuel_type = request.form['fuel_type']
            is_imported = int(request.form['is_imported'])
            transmission = request.form['transmission']
            mileage_raw = request.form['mileage'].replace(',', '').strip()
            try:
                mileage = int(mileage_raw)
                if mileage < 0:
                    return render_template('error.html', error="Số km đã đi không được âm.")
            except Exception:
                return render_template('error.html', error="Số km đã đi không hợp lệ. Vui lòng nhập số nguyên >= 0.")

            # Tính tuổi xe
            current_year = datetime.now().year
            car_age = current_year - year

            # Hàm bin mileage giống như trong train_price_model.py
            def mileage_bin(km):
                if pd.isna(km):
                    return 'unknown'
                km = float(km)
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

            mileage_km_binned = mileage_bin(mileage)

            # Chuẩn bị dữ liệu đầu vào cho mô hình
            input_data = pd.DataFrame({
                'year': [year],
                'engine_capacity': [engine_capacity],
                'car_age': [car_age],
                'is_imported': [is_imported],
                'brand': [brand],
                'body_type': [body_type],
                'fuel_type': [fuel_type],
                'mileage_km': [mileage_km_binned],
                'transmission': [transmission]
            })

            # Nạp mô hình
            model = load_model()
            if model is None:
                return render_template('error.html', error="Không thể nạp mô hình dự đoán. Vui lòng thử lại sau.")

            # Dự đoán giá xe
            predicted_price = model.predict(input_data)[0]

            # Điều chỉnh giá theo loại hộp số
            transmission_lower = transmission.strip().lower()
            if transmission_lower in ["at", "số tự động", "automatic", "auto"]:
                predicted_price += 50  # Cộng 50 triệu cho số tự động
            elif transmission_lower in ["mt", "số sàn", "manual"]:
                predicted_price -= 40  # Trừ 40 triệu cho số sàn

            # Điều chỉnh giá theo số km đã chạy
            # Điều chỉnh giá theo số km đã chạy
            luxury_brands = ["mercedes-benz", "bmw", "audi", "lexus"]
            truck_keywords = ["tải", "truck", "van", "pickup", "dịch vụ"]

            brand_lower = brand.strip().lower()
            body_lower = body_type.strip().lower()
            is_truck = any(x in brand_lower or x in body_lower for x in truck_keywords)

            # Xác định phần trăm giảm theo phân khúc
            if any(b in brand_lower for b in luxury_brands):
                percents = [0.10, 0.18, 0.22, 0.25, 0.28]
            elif is_truck:
                percents = [0.15, 0.25, 0.30, 0.35, 0.38]
            else:
                percents = [0.07, 0.13, 0.17, 0.20, 0.22]

            # Xác định phần trăm giảm tương ứng theo km
            if mileage <= 10000:
                percent = percents[0]
            elif mileage <= 50000:
                percent = percents[1]
            elif mileage <= 100000:
                percent = percents[2]
            elif mileage <= 200000:
                percent = percents[3]
            else:
                percent = 0.7

            # Giảm trực tiếp trên giá dự đoán
            predicted_price = predicted_price * (1 - percent)
            km_note = ""


            # Xử lý kết quả dự đoán
            if predicted_price < 0:
                predicted_price = 0
                confidence = "Thấp (mô hình cho kết quả âm)"
            else:
                confidence_score = 85 - (car_age * 3)
                if not transmission:
                    confidence_score -= 5
                if mileage <= 0:
                    confidence_score -= 5
                if confidence_score >= 80:
                    confidence = "Cao"
                elif confidence_score >= 60:
                    confidence = "Trung bình"
                else:
                    confidence = "Thấp"

            # Format giá dự đoán
            if predicted_price >= 1000:
                billions = int(predicted_price // 1000)
                millions = int(predicted_price % 1000)
                formatted_price = f"{billions} tỷ {millions} triệu VND {km_note}"
            else:
                formatted_price = f"{int(predicted_price)} triệu VND {km_note}"

            mileage_warning = None
            if mileage == 0:
                mileage_warning = "Cảnh báo: Bạn nhập số km đã đi = 0. Giá dự đoán có thể không chính xác vì dữ liệu huấn luyện đã loại bỏ các xe đi 0 km."

            return render_template('predict.html',
                                  brand=brand,
                                  year=year,
                                  body_type=body_type,
                                  engine_capacity=engine_capacity,
                                  fuel_type=fuel_type,
                                  is_imported="Nhập khẩu" if is_imported == 1 else "Trong nước",
                                  transmission=transmission,
                                  mileage=f"{int(mileage):,d} km",
                                  predicted_price=formatted_price,
                                  confidence=confidence,
                                  raw_price=int(predicted_price),
                                  mileage_warning=mileage_warning)
        except Exception as e:
            return render_template('error.html', error=f"Lỗi khi dự đoán: {str(e)}")
    return render_template('predict.html')

@app.route('/visualization')
def visualization():
    """Trang trực quan hóa dữ liệu"""
    try:
        df = load_data()
        if df.empty:
            return render_template(
                'error.html',
                error="Không thể nạp dữ liệu xe để trực quan hóa. Hãy kiểm tra lại file CSV hoặc JSON trong thư mục data/processed."
            )

        # Xác định cột cần dùng phù hợp với dữ liệu thực tế
        price_field = 'price'
        brand_field = 'brand'
        body_type_field = 'body_type'

        # Nếu không có các cột chính, báo lỗi
        missing = [c for c in [price_field, brand_field, body_type_field] if c not in df.columns]
        if missing:
            return render_template('error.html',
                error=f"Dữ liệu bị thiếu các cột cần thiết: {', '.join(missing)}")

        # Tính toán thống kê
        total_cars = len(df)
        top_brand = df[brand_field].value_counts().idxmax()
        top_brand_count = df[brand_field].value_counts().max()

        top_body_type = df[body_type_field].value_counts().idxmax()
        top_body_type_count = df[body_type_field].value_counts().max()


        # Chuyển đổi giá trị price sang số nếu là chuỗi
        def parse_price(x):
            if pd.isnull(x):
                return None
            if isinstance(x, (int, float)):
                return x
            s = str(x).lower().replace(' ', '')
            if 'tỷ' in s:
                parts = s.split('tỷ')
                ty = float(parts[0]) if parts[0] else 0
                trieu = 0
                if len(parts) > 1 and 'triệu' in parts[1]:
                    trieu_part = parts[1].replace('triệu','')
                    trieu = float(trieu_part) if trieu_part else 0
                return ty * 1000 + trieu
            elif 'triệu' in s:
                return float(s.replace('triệu',''))
            elif 'trieu' in s:
                return float(s.replace('trieu',''))
            else:
                try:
                    return float(s)
                except:
                    return None

        price_numeric = df[price_field].apply(parse_price)
        avg_price = price_numeric.mean()
        max_price = price_numeric.max()
        min_price = price_numeric.min()

        # Định dạng giá
        def format_price(price):
            if price is None:
                return "N/A"
            if price >= 1000:
                billions = int(price // 1000)
                millions = int(price % 1000)
                return f"{billions} tỷ {millions} triệu VND"
            else:
                return f"{int(price)} triệu VND"

        stats = {
            'total_cars': total_cars,
            'top_brand': top_brand,
            'top_brand_count': top_brand_count,
            'top_body_type': top_body_type,
            'top_body_type_count': top_body_type_count,
            'avg_price': format_price(avg_price),
            'max_price': format_price(max_price),
            'min_price': format_price(min_price)
        }

        print("Dữ liệu thống kê:", stats)
        return render_template('visualization.html', stats=stats)

    except Exception as e:
        print(" Lỗi visualization:", e)
        return render_template('error.html',
            error=f"Lỗi khi tải trang trực quan hóa: {str(e)}")

@app.route('/classify', methods=['GET', 'POST'])
def classify():
    """Trang phân loại phân khúc xe"""
    if request.method == 'POST':
        import sys
        print("[DEBUG] POST data:", file=sys.stderr)
        print(dict(request.form), file=sys.stderr)
        try:
            # Nhận dữ liệu từ form
            brand = request.form['brand']
            year = int(request.form['year'])
            body_type = request.form['body_type']
            engine_capacity = float(request.form['engine_capacity'])
            features = request.form.getlist('features')  # Danh sách các tính năng đã chọn
            print(f"[DEBUG] Parsed: brand={brand}, year={year}, body_type={body_type}, engine_capacity={engine_capacity}, features={features}", file=sys.stderr)
            # Tính tuổi xe
            current_year = datetime.now().year
            car_age = current_year - year
            print(f"[DEBUG] car_age={car_age}, current_year={current_year}", file=sys.stderr)
            # Tính điểm tính năng (mỗi tính năng đóng góp 1 điểm)
            feature_score = len(features)
            # Phân loại phân khúc xe dựa trên các thông số
            # Logic phân loại đơn giản
            segment = ''
            confidence = 0
            price_estimation = 0
            # Brands cao cấp
            luxury_brands = ['Mercedes-Benz', 'BMW', 'Lexus', 'Audi']
            premium_brands = ['Toyota', 'Honda', 'Mazda']
            # Tạo hệ số điều chỉnh giá theo body_type
            body_type_factor = {
                'SUV': 1.25,
                'Sedan': 1.10,
                'Crossover': 1.15,
                'Hatchback': 0.95,
                'MPV': 1.05,
                'Pickup': 1.08
            }
            bt_factor = body_type_factor.get(body_type, 1.0)

            if brand in luxury_brands:
                if engine_capacity > 2.5 or car_age < 3:
                    segment = 'Luxury'
                    confidence = 90
                    base_price = 1500
                    price_estimation = (base_price + (engine_capacity * 200) - (car_age * 50) + (feature_score * 30)) * bt_factor
                else:
                    segment = 'Premium'
                    confidence = 85
                    base_price = 800
                    price_estimation = (base_price + (engine_capacity * 100) - (car_age * 40) + (feature_score * 20)) * bt_factor
            elif brand in premium_brands or engine_capacity > 2.0 or body_type in ['SUV']:
                segment = 'Premium' if (engine_capacity > 1.8 or car_age < 5) else 'Mid-range'
                confidence = 75 if segment == 'Premium' else 80
                if segment == 'Premium':
                    base_price = 800
                    price_estimation = (base_price + (engine_capacity * 100) - (car_age * 40) + (feature_score * 20)) * bt_factor
                else:
                    base_price = 500
                    price_estimation = (base_price + (engine_capacity * 80) - (car_age * 30) + (feature_score * 15)) * bt_factor
            else:
                if engine_capacity > 1.5 or body_type in ['Sedan', 'Crossover']:
                    segment = 'Mid-range'
                    confidence = 80
                    base_price = 500
                    price_estimation = (base_price + (engine_capacity * 80) - (car_age * 30) + (feature_score * 15)) * bt_factor
                else:
                    segment = 'Economy'
                    confidence = 85
                    base_price = 300
                    price_estimation = (base_price + (engine_capacity * 60) - (car_age * 20) + (feature_score * 10)) * bt_factor
            # Đảm bảo giá không âm
            price_estimation = max(0, price_estimation)
            # Format giá
            if price_estimation >= 1000:
                billions = int(price_estimation // 1000)
                millions = int(price_estimation % 1000)
                formatted_price = f"{billions} tỷ {millions} triệu VND"
            else:
                formatted_price = f"{int(price_estimation)} triệu VND"
            # Dữ liệu để hiển thị các xe tương tự từ cùng phân khúc
            # Trong thực tế, điều này nên được truy vấn từ cơ sở dữ liệu
            similar_models = []
            # Đọc dữ liệu xe từ file json
            df = load_data()
            if not df.empty:
                # Xác định trường chứa thông tin phân khúc giá
                if 'phan_khuc_gia' in df.columns:
                    segment_field = 'phan_khuc_gia'
                elif 'price_segment' in df.columns:
                    segment_field = 'price_segment'
                else:
                    segment_field = None
                if segment_field:
                    # Ánh xạ phân khúc
                    segment_mapping = {
                        'Luxury': ['Sang trọng', 'Luxury'],
                        'Premium': ['Cao cấp', 'Premium'],
                        'Mid-range': ['Trung cấp', 'Mid-range'],
                        'Economy': ['Phổ thông', 'Economy']
                    }
                    # Lọc xe tương tự từ cùng phân khúc
                    segment_values = segment_mapping.get(segment, [segment])
                    similar_cars = df[df[segment_field].isin(segment_values)]
                    # Lấy mẫu 3 xe tương tự
                    if len(similar_cars) > 0:
                        sample_cars = similar_cars.sample(min(3, len(similar_cars)))
                        for _, car in sample_cars.iterrows():
                            model_info = {
                                'brand': car.get('hang_xe', car.get('brand', 'Unknown')),
                                'model': car.get('ten_xe', car.get('name', 'Unknown Model')),
                                'segment': segment
                            }
                            similar_models.append(model_info)
                # Nếu không có trường phân khúc, bỏ qua phần xe tương tự
            # Trả về kết quả dưới dạng JSON nếu là API request
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({
                    'segment': segment,
                    'confidence': confidence,
                    'price_estimation': int(price_estimation),
                    'formatted_price': formatted_price,
                    'similar_models': similar_models
                })
            # Nếu không phải API request, render template với kết quả
            return render_template('classify_result.html', 
                                  brand=brand,
                                  year=year,
                                  body_type=body_type,
                                  engine_capacity=engine_capacity,
                                  features=features,
                                  segment=segment,
                                  confidence=confidence,
                                  price_estimation=int(price_estimation),
                                  formatted_price=formatted_price,
                                  similar_models=similar_models)
        except Exception as e:
            import traceback
            print("[ERROR] Exception in classify():", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({"error": str(e)}), 400
            return render_template('error.html', error=f"Lỗi khi phân loại: {str(e)}")
    
    # GET request - hiển thị form
    return render_template('classify.html', current_year=datetime.now().year)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint cho dự đoán giá xe"""
    try:
        # Nhận dữ liệu từ request JSON
        data = request.get_json()
        
        # Nạp mô hình
        model = load_model()
        if model is None:
            return jsonify({"error": "Không thể nạp mô hình"}), 500
        
        # Tính tuổi xe
        current_year = datetime.now().year
        car_age = current_year - data['year']
        
        # Chuẩn bị dữ liệu đầu vào
        input_data = pd.DataFrame({
            'year': [data['year']],
            'engine_capacity': [data['engine_capacity']],
            'car_age': [car_age],
            'is_imported': [data['is_imported']],
            'brand': [data['brand']],
            'body_type': [data['body_type']],
            'fuel_type': [data['fuel_type']]
        })
        
        # Thêm các trường tùy chọn nếu có
        optional_fields = ['mileage', 'transmission']
        for field in optional_fields:
            if field in data:
                input_data[field] = data[field]
        
        # Dự đoán giá xe
        predicted_price = model.predict(input_data)[0]
        
        # Xử lý kết quả
        if predicted_price < 0:
            predicted_price = 0
        
        # Tính mức độ tin cậy
        confidence_score = 85 - (car_age * 3)
        if 'transmission' not in data or not data['transmission']:
            confidence_score -= 5
        if 'mileage' not in data or data['mileage'] <= 0:
            confidence_score -= 5
            
        confidence = "Cao" if confidence_score >= 80 else "Trung bình" if confidence_score >= 60 else "Thấp"
        
        # Trả về kết quả dự đoán
        return jsonify({
            "predicted_price": int(predicted_price),
            "formatted_price": f"{int(predicted_price)} triệu VND" if predicted_price < 1000 else f"{int(predicted_price//1000)} tỷ {int(predicted_price%1000)} triệu VND",
            "confidence": confidence,
            "confidence_score": confidence_score
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/data/brand_distribution')
def api_brand_distribution():
    """API endpoint cho phân bố thương hiệu"""
    try:
        df = load_data()
        if df.empty:
            return jsonify({"error": "Không thể nạp dữ liệu"}), 500
        
        # Xác định trường chứa thông tin thương hiệu
        brand_field = 'brand' if 'brand' in df.columns else 'hang_xe'
            
        if brand_field in df.columns:
            brand_counts = df[brand_field].value_counts().head(10)
            
            return jsonify({
                "labels": brand_counts.index.tolist(),
                "data": brand_counts.values.tolist()
            })
        else:
            return jsonify({"error": f"Không tìm thấy trường {brand_field} trong dữ liệu"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/data/body_type_distribution')
def api_body_type_distribution():
    """API endpoint cho phân bố kiểu dáng xe"""
    try:
        df = load_data()
        if df.empty:
            return jsonify({"error": "Không thể nạp dữ liệu"}), 500
        
        # Xác định trường chứa thông tin kiểu dáng
        body_type_field = 'body_type' if 'body_type' in df.columns else 'kieu_dang'
            
        if body_type_field in df.columns:
            body_type_counts = df[body_type_field].value_counts()
            
            return jsonify({
                "labels": body_type_counts.index.tolist(),
                "data": body_type_counts.values.tolist()
            })
        else:
            return jsonify({"error": f"Không tìm thấy trường {body_type_field} trong dữ liệu"}), 400
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/data/price_by_age')
def api_price_by_age():
    """API endpoint cho tương quan giữa tuổi xe và giá bán"""
    try:
        df = load_data()
        if df.empty:
            return jsonify({"error": "Không thể nạp dữ liệu"}), 500
        
        # Xác định các trường cần thiết
        price_field = 'gia_trieu' if 'gia_trieu' in df.columns else 'price_million'
        age_field = 'tuoi_xe' if 'tuoi_xe' in df.columns else 'car_age'
        brand_field = 'hang_xe' if 'hang_xe' in df.columns else 'brand'

        # Nếu price_field chưa có, tạo từ cột 'price' (nếu có)
        if price_field not in df.columns:
            if 'price' in df.columns:
                df['price_million'] = df['price'].apply(parse_price_text_to_million)
                price_field = 'price_million'
            else:
                return jsonify({"error": f"Không tìm thấy trường giá ('{price_field}' hoặc 'price') trong dữ liệu"}), 400

        # Nếu age_field chưa có, tạo từ cột 'year' (nếu có)
        if age_field not in df.columns:
            if 'year' in df.columns:
                current_year = datetime.now().year
                df['car_age'] = current_year - df['year']
                age_field = 'car_age'
            else:
                return jsonify({"error": f"Không tìm thấy trường tuổi xe ('{age_field}' hoặc 'year') trong dữ liệu"}), 400

        # Nếu brand_field chưa có, báo lỗi
        if brand_field not in df.columns:
            return jsonify({"error": f"Không tìm thấy trường thương hiệu ('{brand_field}') trong dữ liệu"}), 400

        # Lọc các giá trị không hợp lệ
        df_filtered = df[(df[price_field].notnull()) & (df[age_field].notnull())]

        # Lấy 5 thương hiệu phổ biến nhất
        top_brands = df_filtered[brand_field].value_counts().head(5).index.tolist()

        # Dữ liệu cho scatter plot
        result = []
        for brand in top_brands:
            brand_data = df_filtered[df_filtered[brand_field] == brand]
            points = []

            for _, row in brand_data.iterrows():
                points.append({
                    "x": float(row[age_field]),
                    "y": float(row[price_field])
                })

            result.append({
                "label": brand,
                "data": points
            })

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/data/avg_price_by_body_type')
def api_avg_price_by_body_type():
    """API endpoint cho giá trung bình theo kiểu dáng xe"""
    try:
        df = load_data()
        if df.empty:
            return jsonify({"error": "Không thể nạp dữ liệu"}), 500
        
        # Xác định các trường cần thiết
        price_field = 'gia_trieu' if 'gia_trieu' in df.columns else 'price_million'
        body_type_field = 'body_type' if 'body_type' in df.columns else 'kieu_dang'

        # Nếu price_field chưa có, tạo từ cột 'price' (nếu có)
        if price_field not in df.columns:
            if 'price' in df.columns:
                df['price_million'] = df['price'].apply(parse_price_text_to_million)
                price_field = 'price_million'
            else:
                return jsonify({"error": f"Không tìm thấy trường giá ('{price_field}' hoặc 'price') trong dữ liệu"}), 400

        # Nếu body_type_field chưa có, báo lỗi
        if body_type_field not in df.columns:
            return jsonify({"error": f"Không tìm thấy trường kiểu dáng ('{body_type_field}') trong dữ liệu"}), 400

        # Tính giá trung bình theo kiểu dáng
        avg_price = df.groupby(body_type_field)[price_field].mean().reset_index()

        # Sắp xếp theo giá trị giảm dần
        avg_price = avg_price.sort_values(price_field, ascending=False)

        return jsonify({
            "labels": avg_price[body_type_field].tolist(),
            "data": [round(x) for x in avg_price[price_field].tolist()]
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/data/price_trends')
def api_price_trends():
    """API endpoint cho xu hướng giá xe theo thời gian"""
    try:
        df = load_data()
        if df.empty:
            return jsonify({"error": "Không thể nạp dữ liệu"}), 500
        
        # Xác định các trường cần thiết
        price_field = 'gia_trieu' if 'gia_trieu' in df.columns else 'price_million'
        year_field = 'nam_sx' if 'nam_sx' in df.columns else 'year'
        body_type_field = 'body_type' if 'body_type' in df.columns else 'kieu_dang'

        # Nếu price_field chưa có, tạo từ cột 'price' (nếu có)
        if price_field not in df.columns:
            if 'price' in df.columns:
                df['price_million'] = df['price'].apply(parse_price_text_to_million)
                price_field = 'price_million'
            else:
                return jsonify({"error": f"Không tìm thấy trường giá ('{price_field}' hoặc 'price') trong dữ liệu"}), 400

        # Nếu year_field chưa có, báo lỗi
        if year_field not in df.columns:
            return jsonify({"error": f"Không tìm thấy trường năm sản xuất ('{year_field}') trong dữ liệu"}), 400

        # Nếu body_type_field chưa có, báo lỗi
        if body_type_field not in df.columns:
            return jsonify({"error": f"Không tìm thấy trường kiểu dáng ('{body_type_field}') trong dữ liệu"}), 400

        # Lọc các năm gần đây (5-10 năm gần nhất)
        current_year = datetime.now().year
        recent_years = [str(year) for year in range(current_year - 10, current_year + 1)]

        # Chuyển năm sản xuất sang chuỗi để so sánh
        df[year_field] = df[year_field].astype(str)
        df_recent = df[df[year_field].isin(recent_years)]

        # Lấy 3 kiểu dáng phổ biến nhất
        top_body_types = df[body_type_field].value_counts().head(3).index.tolist()

        # Chuẩn bị dữ liệu
        result = []
        years = sorted(df_recent[year_field].unique().tolist(), key=int)

        for body_type in top_body_types:
            data_points = []
            body_type_df = df_recent[df_recent[body_type_field] == body_type]

            for year in years:
                year_data = body_type_df[body_type_df[year_field] == year]
                if not year_data.empty:
                    avg_price = year_data[price_field].mean()
                    data_points.append(round(avg_price))
                else:
                    data_points.append(None)  # Không có dữ liệu cho năm này

            result.append({
                "label": body_type,
                "data": data_points
            })

        return jsonify({
            "labels": years,
            "datasets": result
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/data/brand_distribution')
def brand_distribution():
    """API cung cấp phân bố thương hiệu xe"""
    try:
        df = load_data()
        brand_field = 'hang_xe' if 'hang_xe' in df.columns else 'brand'
        
        if brand_field not in df.columns:
            return jsonify({"error": f"Không tìm thấy trường {brand_field} trong dữ liệu"})
        
        # Lấy top 10 thương hiệu phổ biến nhất
        brand_counts = df[brand_field].value_counts().head(10)
        
        # Định dạng để trả về
        data = {
            "labels": brand_counts.index.tolist(),
            "data": brand_counts.values.tolist()
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/data/price_distribution')
def price_distribution():
    """API cung cấp phân bố giá xe"""
    try:
        df = load_data()
        # Chuyển đổi giá sang số triệu
        def parse_price_text_to_million(price_str):
            if pd.isnull(price_str):
                return None
            s = str(price_str).lower().replace(' ', '')
            if 'tỷ' in s:
                parts = s.split('tỷ')
                ty = float(parts[0]) if parts[0] else 0
                trieu = 0
                if len(parts) > 1 and 'triệu' in parts[1]:
                    trieu_part = parts[1].replace('triệu','')
                    trieu = float(trieu_part) if trieu_part else 0
                return ty * 1000 + trieu
            elif 'triệu' in s:
                return float(s.replace('triệu',''))
            elif 'trieu' in s:
                return float(s.replace('trieu',''))
            else:
                try:
                    return float(s)
                except:
                    return None

        df['price_million'] = df['price'].apply(parse_price_text_to_million)
        price_field = 'price_million'
        if price_field not in df.columns:
            return jsonify({"error": f"Không tìm thấy trường {price_field} trong dữ liệu"})
        # Tạo các khoảng giá
        price_ranges = [0, 200, 400, 600, 800, 1000, 1500, 2000, 3000, float('inf')]
        price_labels = ['0-200', '200-400', '400-600', '600-800', '800-1000', 
                      '1000-1500', '1500-2000', '2000-3000', '3000+']
        # Phân loại giá theo khoảng
        price_bins = pd.cut(df[price_field], bins=price_ranges, labels=price_labels)
        price_counts = price_bins.value_counts().sort_index()
        # Định dạng để trả về
        data = {
            "labels": price_counts.index.tolist(),
            "data": price_counts.values.tolist()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/data/year_price_relation')
def year_price_relation():
    """API cung cấp mối quan hệ giữa năm sản xuất và giá"""
    try:
        df = load_data()
        # Chuyển đổi giá sang số triệu
        def parse_price_text_to_million(price_str):
            if pd.isnull(price_str):
                return None
            s = str(price_str).lower().replace(' ', '')
            if 'tỷ' in s:
                parts = s.split('tỷ')
                ty = float(parts[0]) if parts[0] else 0
                trieu = 0
                if len(parts) > 1 and 'triệu' in parts[1]:
                    trieu_part = parts[1].replace('triệu','')
                    trieu = float(trieu_part) if trieu_part else 0
                return ty * 1000 + trieu
            elif 'triệu' in s:
                return float(s.replace('triệu',''))
            elif 'trieu' in s:
                return float(s.replace('trieu',''))
            else:
                try:
                    return float(s)
                except:
                    return None

        df['price_million'] = df['price'].apply(parse_price_text_to_million)
        price_field = 'price_million'
        year_field = 'year'
        if price_field not in df.columns or year_field not in df.columns:
            return jsonify({"error": f"Không tìm thấy trường dữ liệu cần thiết trong dữ liệu"})
        # Nhóm theo năm và tính giá trung bình
        year_price_data = df.groupby(year_field)[price_field].mean().reset_index()
        year_price_data = year_price_data.sort_values(by=year_field)
        # Chỉ lấy dữ liệu từ năm 2000 trở lên
        year_price_data = year_price_data[year_price_data[year_field] >= 2000]
        # Định dạng để trả về
        data = {
            "labels": year_price_data[year_field].tolist(),
            "data": year_price_data[price_field].tolist()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/data/mileage_price_relation')
def mileage_price_relation():
    """API cung cấp mối quan hệ giữa số km đã đi và giá"""
    try:
        df = load_data()
        # Chuyển đổi giá sang số triệu
        def parse_price_text_to_million(price_str):
            if pd.isnull(price_str):
                return None
            s = str(price_str).lower().replace(' ', '')
            if 'tỷ' in s:
                parts = s.split('tỷ')
                ty = float(parts[0]) if parts[0] else 0
                trieu = 0
                if len(parts) > 1 and 'triệu' in parts[1]:
                    trieu_part = parts[1].replace('triệu','')
                    trieu = float(trieu_part) if trieu_part else 0
                return ty * 1000 + trieu
            elif 'triệu' in s:
                return float(s.replace('triệu',''))
            elif 'trieu' in s:
                return float(s.replace('trieu',''))
            else:
                try:
                    return float(s)
                except:
                    return None

        df['price_million'] = df['price'].apply(parse_price_text_to_million)
        price_field = 'price_million'
        mileage_field = 'mileage_km'
        if price_field not in df.columns or mileage_field not in df.columns:
            return jsonify({"error": f"Không tìm thấy trường dữ liệu cần thiết trong dữ liệu"})
        # Lọc dữ liệu hợp lệ
        filtered_df = df[(df[mileage_field] > 0) & (df[mileage_field] < 500000)]
        # Tạo các khoảng số km
        mileage_ranges = [0, 10000, 30000, 50000, 70000, 100000, 150000, 200000, float('inf')]
        mileage_labels = ['0-10k', '10k-30k', '30k-50k', '50k-70k', '70k-100k', '100k-150k', '150k-200k', '200k+']
        # Nhóm theo khoảng số km và tính giá trung bình
        filtered_df['mileage_bin'] = pd.cut(filtered_df[mileage_field], bins=mileage_ranges, labels=mileage_labels)
        mileage_price_data = filtered_df.groupby('mileage_bin')[price_field].mean().reset_index()
        # Định dạng để trả về
        data = {
            "labels": mileage_price_data['mileage_bin'].tolist(),
            "data": mileage_price_data[price_field].tolist()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/data/body_type_distribution')
def body_type_distribution():
    """API cung cấp phân bố kiểu dáng xe"""
    try:
        df = load_data()
        body_type_field = 'kieu_dang' if 'kieu_dang' in df.columns else 'body_type'
        
        if body_type_field not in df.columns:
            return jsonify({"error": f"Không tìm thấy trường {body_type_field} trong dữ liệu"})
            
        # Đếm số lượng của mỗi kiểu dáng
        body_type_counts = df[body_type_field].value_counts()
        
        # Định dạng để trả về
        data = {
            "labels": body_type_counts.index.tolist(),
            "data": body_type_counts.values.tolist()
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    # print(app.url_map)
    app.run(debug=True)
    


