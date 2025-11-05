# Dự đoán giá xe ô tô cũ đã qua sử dụng tại Hà Nội


Dự án này xây dựng các mô hình học máy để phân tích, phân loại và dự đoán giá xe ô tô cũ tại Việt Nam dựa trên bộ dữ liệu từ Bonbanh.com. Hệ thống sử dụng các đặc điểm quan trọng của xe như thương hiệu, năm sản xuất, số km đã đi, loại động cơ, hộp số, v.v. không chỉ để dự đoán giá trị mà còn phân loại xe vào các phân khúc giá khác nhau.

## Mục đích

- Xây dựng mô hình hồi quy có khả năng dự đoán chính xác giá trị thị trường của xe ô tô đã qua sử dụng
- Phân loại xe vào các phân khúc giá khác nhau (Thấp, Trung bình, Cao, Cao cấp) dựa trên đặc điểm kỹ thuật
- Phân tích mối tương quan giữa các đặc điểm của xe và giá bán
- Tạo các công cụ trực quan hóa để hiểu rõ hơn về thị trường xe cũ
- Hỗ trợ người mua và người bán trong việc đưa ra quyết định tài chính tốt hơn
## Thông tin bổ sung
https://bonbanh.com/ha-noi/oto-cu-da-qua-su-dung

# Tác giả

| Họ và tên          | Mã sinh viên | Tên GitHub     | Đóng góp   |
|--------------------|--------------|----------------|-------------|
| Nguyễn Thái Dương  | 23001859     | Duong5326      | Đóng góp 1 |
| Lê Khả Dũng        | 23001847     | github_name    | Đóng góp 2 |
| Nguyễn Hữu Duy     | 23001843     | 23001853-wq    | Đóng góp 3 |


## Cấu trúc dự án

```
MachineLearning/
├── data/
│   ├── raw/                            # Dữ liệu gốc
│   │   ├── used_cars.csv              # Dataset chính từ Bonbanh.com
│   │   ├── bot.py & bot2.py           # Scripts crawl data
│   └── processed/                      # Dữ liệu đã xử lý
│       ├── enhanced_car_data.csv      # Dữ liệu cuối cùng (8 features)
│       ├── clean_raw_to_processed.py  # Script làm sạch
│       ├── enhance_car_data.py        # Feature engineering
│       └── models/                    # Trained Models
│           ├── RandomForest_model.pkl        # Regression model
│           ├── Random_Forest_classifier.pkl  # Classification model
│           └── best_model_name.txt          # Model metadata
├── notebooks/
│   └── 08_comprehensive_training.ipynb # Training pipeline hoàn chỉnh
├── templates/                          # HTML Templates
│   ├── index.html                     # Trang chủ - form dự đoán giá
│   ├── predict.html                   # Kết quả dự đoán
│   ├── classify.html                  # Form phân loại phân khúc
│   ├── classify_result.html           # Kết quả phân loại
│   ├── visualization.html             # Dashboard trực quan
│   ├── layout.html                    # Base template
│   └── error.html                     # Error handling
├── static/css/
│   └── style.css                      # Bootstrap custom styles
├── application.py                     # Ứng dụng Flask
├── requirements.txt                   # Danh sách thư viện phụ thuộc
├── README.md                          # Tài liệu chính
└── Other docs/                        # Báo cáo và hướng dẫn
    ├── README_BAI_TAP_LON.md
    ├── PROJECT_SUMMARY.md
    ├── STRUCTURE_REPORT.md
    └── ...
```

## Mô hình Machine Learning

### Kiến trúc hệ thống
- **RandomForest Regression** - Dự đoán giá xe với R² Score cao
- **RandomForest Classification** - Phân loại 4 phân khúc (91.6% accuracy)
- **Flask Web App** - Giao diện người dùng với Bootstrap + Chart.js
- **Feature Engineering** - 8 đặc trưng được tối ưu hóa

### Dataset và Performance
- **Nguồn**: Bonbanh.com (thị trường xe cũ Hà Nội)
- **Kích thước**: 13,500+ xe ô tô đã qua sử dụng
- **Features**: 8 đặc trưng sau khi loại bỏ multicollinearity

| Model | Algorithm | Performance | Mục đích |
|-------|-----------|-------------|----------|
| **Regression** | RandomForest | High R² Score | Dự đoán giá chính xác |
| **Classification** | RandomForest | 91.6% accuracy | Phân loại Economy/Mid/Premium/Luxury |

### 8 đặc trưng cuối cùng
1. **engine_capacity** - Dung tích động cơ (L)
2. **car_age** - Tuổi xe (năm) - thay thế 'year' để fix multicollinearity
3. **origin** - Xuất xứ (Lắp ráp trong nước/Nhập khẩu)
4. **brand** - Thương hiệu (Toyota, Honda, BMW...)
5. **body_type** - Kiểu dáng (Sedan, SUV, Hatchback...)
6. **fuel_type** - Loại nhiên liệu (Xăng, Dầu)
7. **mileage_km** - Số km đã đi
8. **transmission** - Hộp số (Số tự động/Số sàn)

### Phân tích độ nhạy đặc trưng
- **Origin**: +294 triệu VND (Nhập khẩu vs Trong nước)
- **Transmission**: +16.7 triệu VND (Tự động vs Sàn)
- **Mileage**: Tương quan thực tế với thị trường
- **Brand & Body Type**: Tác động đáng kể đến giá

## Hướng dẫn sử dụng

### Cài đặt và chạy
```bash
# 1. Clone repository
git clone https://github.com/Duong5326/MachineLearning.git
cd MachineLearning

# 2. Cài đặt dependencies
pip install -r requirements.txt

# 3. Chạy ứng dụng
python application.py
```
**Truy cập**: http://localhost:5000

### Chức năng chính
| Trang | URL | Mô tả |
|-------|-----|-------|
| **Trang chủ** | `/` | Form dự đoán giá xe |
| **Kết quả** | `/predict` | Hiển thị giá dự đoán và thông tin |
| **Phân loại** | `/classify` | Phân loại phân khúc xe |
| **Trực quan** | `/visualization` | Biểu đồ và thống kê |

### Demo
**Input mẫu**: Toyota, 2020, 2.0L, Sedan, 50,000km, Trong nước, Tự động, Xăng  
**Output**: ~750 triệu VND

**Test sensitivity**:
- Đổi "Trong nước" → "Nhập khẩu": +294 triệu VND
- Đổi "Số sàn" → "Tự động": +16.7 triệu VND

## Training Models (Tùy chọn)
Models đã được train sẵn. Nếu cần train lại:
```bash
jupyter notebook notebooks/08_comprehensive_training.ipynb
```
Pipeline sẽ thực hiện:
1. Load & preprocess data
2. Train RandomForest regression
3. Train RandomForest classification
4. Save models to `data/processed/models/`

## Kỹ thuật và công nghệ

### Tech Stack
**Backend**: Flask 2.3.3, Python 3.8+  
**ML Libraries**: scikit-learn, pandas, numpy, joblib  
**Frontend**: Bootstrap 5, Chart.js, AJAX/jQuery  
**Visualization**: matplotlib, seaborn, plotly  

### Tối ưu hóa
- **Class-based Architecture**: OOP design pattern
- **Caching Strategy**: Global model & data caching  
- **Error Handling**: Comprehensive exception management
- **Lazy Loading**: Models loaded on-demand

### Đánh giá mô hình
**Regression**: R² Score, RMSE, MAE, Feature Sensitivity Analysis  
**Classification**: 91.6% Accuracy, Precision, Recall, F1-Score, Confusion Matrix



