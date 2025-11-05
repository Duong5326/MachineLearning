# Tài liệu dữ liệu - Dự án dự đoán giá xe ô tô

## Cấu trúc dữ liệu

### Thư mục `raw/`
Chứa dữ liệu thô từ quá trình thu thập:
- `used_cars.csv`: Dataset chính từ bonbanh.com (13,500+ xe)
- `bot.py` & `bot2.py`: Scripts crawl dữ liệu từ bonbanh.com 
### Thư mục `processed/`
Chứa dữ liệu đã xử lý và models:

- **Dữ liệu chính**:
  - `enhanced_car_data.csv`: Dữ liệu cuối cùng với 8 features tối ưu
  - `enhanced_car_data.json`: Định dạng JSON tương ứng

- **Models đã huấn luyện**:
  - `models/RandomForest_model.pkl`: Regression model (dự đoán giá)
  - `models/Random_Forest_classifier.pkl`: Classification model (phân loại phân khúc)
  - `models/best_model_name.txt`: Metadata model

- **Scripts xử lý**:
  - `clean_raw_to_processed.py`: Làm sạch và chuẩn hóa dữ liệu từ raw
  - `enhance_car_data.py`: Feature engineering, tách engine info, tạo categories

## Dataset và Features

### Dữ liệu gốc (`raw/used_cars.csv`)
Dataset chính từ bonbanh.com với các trường:
- Ten_xe, Nam_san_xuat, So_km, Xuat_xu
- Kieu_dang, Hop_so, Dong_co, Gia_ban_text
- Mau_ngoai_that, So_cho_ngoi, Link

### 8 Features cuối cùng (sau feature engineering)
| Feature | Mô tả | Ví dụ |
|---------|-------|--------|
| **engine_capacity** | Dung tích động cơ (L) | 2.0, 1.5, 3.0 |
| **car_age** | Tuổi xe (năm) | 3, 5, 8 |
| **origin** | Xuất xứ | "Lắp ráp trong nước", "Nhập khẩu" |
| **brand** | Thương hiệu | "Toyota", "Honda", "BMW" |
| **body_type** | Kiểu dáng | "Sedan", "SUV", "Hatchback" |
| **fuel_type** | Loại nhiên liệu | "Xăng", "Dầu" |
| **mileage_km** | Số km đã đi | 50000, 100000, 20000 |
| **transmission** | Hộp số | "Số tự động", "Số sàn" |

### Features được thêm mới
- **car_age**: Tính từ năm hiện tại trừ đi năm sản xuất (thay cho year để fix multicollinearity)
- **engine_capacity**: Tách từ trường "Dong_co" ("Xăng 2.0 L" → 2.0)  
- **fuel_type**: Tách từ trường "Dong_co" ("Xăng 2.0 L" → "Xăng")
- **price_million**: Chuyển đổi "Gia_ban_text" từ text ("645 Triệu") sang số (645)
- **price_category**: Phân loại theo giá ("Trung cấp", "Cao cấp", "Sang trọng"...)
- **segment**: Phân khúc chi tiết ("Sedan hạng C", "SUV cỡ lớn"...)

### Features được chuẩn hóa tên
- Ten_xe → name, Nam_san_xuat → year
- Xuat_xu → origin, Kieu_dang → body_type  
- Hop_so → transmission, So_km → mileage_km
- Gia_ban_text → price

## Training Models

Models đã được train sẵn. Để train lại:
```bash
# chạy scripts xử lý dữ liệu
python processed/clean_raw_to_processed.py
python processed/enhance_car_data.py

# Mở notebook training
jupyter notebook notebooks/training.ipynb

```

## Thống kê dữ liệu
- **Dữ liệu gốc**: 13,500+ xe từ bonbanh.com
- **Sau xử lý**: ~ 13500 xe đã làm sạch
- **Features**: 8 đặc trưng tối ưu (đã fix multicollinearity)
- **Performance**: RandomForest 91.6% accuracy cho classification

## Các files quan trọng
- `raw/used_cars.csv`: Dataset gốc từ bonbanh.com
- `processed/enhanced_car_data.csv`: Dữ liệu cuối cùng với 8 features
- `processed/models/RandomForest_model.pkl`: Model dự đoán giá
- `processed/models/Random_Forest_classifier.pkl`: Model phân loại phân khúc