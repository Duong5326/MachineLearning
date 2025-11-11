# Dự đoán giá xe ô tô cũ đã qua sử dụng tại Hà Nội


Dự án này xây dựng các mô hình học máy để phân tích, phân loại và dự đoán giá xe ô tô cũ tại Việt Nam dựa trên bộ dữ liệu từ Bonbanh.com. Hệ thống sử dụng các đặc điểm quan trọng của xe như thương hiệu, năm sản xuất, số km đã đi, loại động cơ, hộp số, v.v. không chỉ để dự đoán giá trị mà còn phân loại xe vào các phân khúc giá khác nhau.

## Mục đích

- Xây dựng mô hình hồi quy có khả năng dự đoán chính xác giá trị thị trường của xe ô tô đã qua sử dụng
- Phân loại xe vào các phân khúc giá khác nhau (Thấp, Trung bình, Cao, Cao cấp) dựa trên đặc điểm kỹ thuật
- Phân tích mối tương quan giữa các đặc điểm của xe và giá bán
- Tạo các công cụ trực quan hóa để hiểu rõ hơn về thị trường xe cũ
- Hỗ trợ người mua và người bán trong việc đưa ra quyết định tài chính tốt hơn

# 3.3. Thông tin thành viên nhóm, công việc của mỗi thành viên 

## Thành viên nhóm
| Thành viên             | MSSV     | Vai trò chính                                   |
|-----------------------|----------|-------------------------------------------------|
| Nguyễn Thái Dương     | 23001859 | Leader, Data Engineer                           |
| Lê Khả Dũng           | 23001847 | ML Engineer (Clustering & Dimensionality Reduction) |
| Nguyễn Hữu Duy        | 23001843 | ML Engineer (Classification), Frontend Developer |

## Chi tiết phân công công việc
| Thành viên             | Công việc thực hiện                                                                 |
|-----------------------|-----------------------------------------------------------------------------------|
| Nguyễn Thái Dương     | - Thu thập, tiền xử lý dữ liệu, xây dựng pipeline xử lý (bot.py, clean_raw_to_processed.py, enhance_car_data.py) |
|                       | - Thực hiện và so sánh các mô hình hồi quy (RandomForest, LinearRegression, Lasso, KNN), phân tích kết quả, giải thích mô hình |
|                       | - Phối hợp với các thành viên trong phân tích, đánh giá, so sánh các phương pháp, viết báo cáo phần dữ liệu và phương pháp |
| Lê Khả Dũng           | - Thực hiện các phương pháp giảm chiều và phân cụm (PCA, KMeans...), phân tích và đánh giá kết quả |
|                       | - Phối hợp tiền xử lý dữ liệu, đánh giá ảnh hưởng của giảm chiều lên mô hình, viết báo cáo phần clustering, giảm chiều |
|                       | - Tham gia hỗ trợ so sánh kết quả giữa các phương pháp trước/sau khi giảm chiều |
| Nguyễn Hữu Duy        | - Xây dựng mô hình RandomForest Classification (phân loại 4 phân khúc giá) |
|                       | - Phát triển Flask web application với Bootstrap UI |
|                       | - Thiết kế giao diện người dùng, integration testing và deployment |
|                       | - Viết báo cáo phần classification và demo ứng dụng |

## Kỹ thuật ML được sử dụng bởi từng thành viên
- Thành viên 1: Regression (RandomForest, Linear, Lasso, KNN), Data Preprocessing, Feature Engineering  
- Thành viên 2: Dimensionality Reduction (PCA), Clustering (KMeans), Data Evaluation  
- Thành viên 3: Classification (RandomForest,KNN), Model Evaluation, Web Development (Flask)

## 3.4. Hướng dẫn tổ chức dữ liệu và kịch bản thực nghiệm

### Link nguồn dữ liệu
- **Nguồn chính**: https://bonbanh.com/ha-noi/oto-cu-da-qua-su-dung
- **Dataset gốc**: `data/raw/used_cars.csv` (~13,500 xe ô tô cũ tại Hà Nội)
- **Dataset đã xử lý**: `data/processed/enhanced_car_data.csv` (8 features tối ưu)

### Cấu trúc tổ chức dữ liệu

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
│           ├── KNN_model.pkl                   # Regression model
│           ├── RandomForest_model.pkl          # Regression model
│           ├── Lasso_model.pkl                 # Regression model
│           ├── LinearRegression_model.pkl      # Regression model
│           ├── Random_Forest_classifier.pkl  # Classification model
│           └── best_model_name.txt          # Model metadata
├── notebooks/
│   └── training.ipynb # Training pipeline hoàn chỉnh
├── templates/                          # HTML Templates
│   ├── index.html                     # Trang chủ - form dự đoán giá
│   ├── predict.html                   # Kết quả dự đoán
│   ├── classify.html                  # Form phân loại phân khúc
│   ├── classify_result.html           # Kết quả phân loại
│   ├── visualization.html             # Dashboard trực quan dùng chart.js 
│   ├── visualization_result.html      # Dashboard trực quan từ huấn luyện
│   ├── layout.html                    # Base template
│   └── error.html                     # Error handling
├── static/css/
│   └── style.css                      # Bootstrap custom styles
│   └── png                            # Ảnh qua huấn luyện
├── application.py                     # Ứng dụng Flask
├── requirements.txt                   # Danh sách thư viện phụ thuộc
└── README.md                          # Tài liệu chính
```

### Kịch bản thực nghiệm chi tiết

#### Bước 1: Chuẩn bị môi trường và dữ liệu
```bash
# Clone repository
git clone https://github.com/Duong5326/MachineLearning.git
cd MachineLearning

# Cài đặt dependencies
pip install -r requirements.txt

# Chạy script crawl dữ liệu (nếu cần thu thập dữ liệu mới)
cd data/raw/
python bot.py
python bot2.py

# Làm sạch và xử lý dữ liệu
cd ../processed/
python clean_raw_to_processed.py    # Vietnamese → English mapping
python enhance_car_data.py          # Feature engineering → 8 features
```

#### Bước 2: Training và đánh giá mô hình
```bash
# Chạy notebook training hoàn chỉnh
jupyter notebook notebooks/training.ipynb

# Hoặc chạy từng bước để phân tích chi tiết:
# 1. Data exploration và visualization
# 2. Multiple algorithms comparison:
#    - LinearRegression, Lasso, KNN, RandomForest (Regression)
#    - RandomForest, KNN (Classification)  
# 3. Cross-validation và hyperparameter tuning
# 4. Model selection based on performance metrics
# 5. Feature importance analysis cho final models
```

#### Bước 3: Kiểm tra các mô hình đã trained
```bash
# Test các regression models (4 variants với train/test ratios khác nhau)
python -c "
from joblib import load
import os
models_dir = 'data/processed/models/'
regression_models = [f for f in os.listdir(models_dir) if 'Random' in f and 'class' not in f]
print(f'Regression models available: {len(regression_models)}')
for model in regression_models:
    print(f'- {model}')
"

# Test classification model (best performer)
python -c "
from joblib import load
classifier = load('data/processed/models/Random_Forest_classifier.pkl')  
print(f'Classification model: {type(classifier)}')
print('Performance: 91.6% accuracy on 4-class classification')
print('Classes: Economy, Mid-range, Premium, Luxury')
"
```

#### Bước 4: Demo web application
```bash
# Chạy Flask app
python application.py

# Truy cập và test:
# - http://localhost:5000/          → Dự đoán giá xe
# - http://localhost:5000/classify  → Phân loại phân khúc  
# - http://localhost:5000/visualization → Dashboard trực quan dùng chart.js 
# - http://localhost:5000/visualization_result → # Dashboard trực quan từ huấn luyện
```

### Hướng dẫn sử dụng file kết quả
- **Regression Models**: `data/processed/models/` - 4 RandomForest regression models với train/test ratios khác nhau
- **Classification Model**: `data/processed/models/Random_Forest_classifier.pkl` - Best performer (91.6% accuracy)
- **Comparison Results**: Tất cả algorithms được so sánh trong notebook (Linear, Lasso, KNN, RF)
- **Data**: `data/processed/enhanced_car_data.csv` - Dataset cuối cùng với 8 features đã optimize
- **Training Process**: Chi tiết model selection và hyperparameter tuning trong notebook
- **Performance Metrics**: Cross-validation scores, feature importance, confusion matrix

## Mô hình Machine Learning

### Quá trình nghiên cứu và lựa chọn mô hình

#### Các thuật toán đã thực nghiệm
**Hồi quy (Regression)**:
- **LinearRegression** - Baseline model
- **Lasso Regression** - Regularization approach  
- **KNeighborsRegressor** - Instance-based learning
- **RandomForestRegressor** - Ensemble method

**Phân loại (Classification)**:
- **RandomForestClassifier** - Ensemble method
- **KNeighborsClassifier** - Instance-based learning

**Tiền xử lý và phân tích**:
- **PCA & TruncatedSVD** - Dimensionality reduction
- **KMeans & DBSCAN** - Clustering analysis
- **StandardScaler** - Feature normalization

#### Kết quả so sánh và lựa chọn mô hình cuối cùng

| Thuật toán | Loại | Performance | Trạng thái | Ghi chú |
|------------|------|-------------|------------|---------|
| **KNeighborsRegressor** | Hồi quy | R² cao nhất | ✅ **Được chọn** | Tốt nhất với dữ liệu đã chuẩn hóa, 8 đặc trưng |
| RandomForestRegressor | Hồi quy | R² cao nhưng thấp hơn KNN | ❌ Không dùng | Phù hợp với dữ liệu nhiều đặc trưng, nhưng KNN tốt hơn với 8 đặc trưng |
| LinearRegression | Hồi quy | R² thấp | ❌ Không dùng | Underfitting với dữ liệu phức tạp |
| Lasso Regression | Hồi quy | R² trung bình | ❌ Không dùng | Over-regularization |
| **RandomForestClassifier** | Phân loại | 91.6% accuracy | ✅ **Được chọn** | Tốt nhất cho 4-class classification |
| KNeighborsClassifier | Phân loại | Accuracy thấp hơn | ❌ Không dùng | Kém hiệu quả với high-dim data |

### Kiến trúc hệ thống cuối cùng
- **KNN Regression** - Dự đoán giá xe (mô hình tốt nhất với 8 đặc trưng, R² cao nhất)
- **RandomForest Classification** - Phân loại 4 phân khúc giá (91.6% accuracy) 
- **Flask Web App** - Giao diện người dùng với Bootstrap + Chart.js
- **Feature Engineering** - 8 đặc trưng được tối ưu hóa

### Dataset và Performance
- **Nguồn**: Bonbanh.com (thị trường xe cũ Hà Nội)
- **Kích thước**: ~13,500 xe ô tô đã qua sử dụng
- **Features**: 8 đặc trưng sau khi loại bỏ multicollinearity

| Model | Algorithm | Performance | Mục đích |
|-------|-----------|-------------|----------|
| **Regression** | KNN | High R² Score | Dự đoán giá chính xác |
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
Cả 8 đặc trưng đều tác động đến giá xe, nhưng mức độ ảnh hưởng **không cố định** mà phụ thuộc vào từng hãng xe, dòng xe, năm sản xuất và các yếu tố thị trường. Các con số như Origin (+294 triệu VND) hay Transmission (+16.7 triệu VND) chỉ là ước lượng trung bình trên toàn bộ dữ liệu, mang tính tham khảo tổng thể. Thực tế, chênh lệch giá này sẽ khác nhau giữa các hãng, dòng xe và từng trường hợp cụ thể.
Ví dụ:
- Xe nhập khẩu của Toyota có thể chênh lệch giá khác so với BMW.
- Hộp số tự động ở xe phổ thông tăng giá ít hơn xe sang.
Do đó, khi dự đoán giá, mô hình sẽ kết hợp đồng thời cả 8 đặc trưng để đưa ra kết quả phù hợp nhất cho từng xe cụ thể.

## Hướng dẫn sử dụng wed

### Chức năng chính
| Trang | URL | Mô tả |
|-------|-----|-------|
| **Trang chủ** | `/` | Form dự đoán giá xe |
| **Kết quả** | `/predict` | Hiển thị giá dự đoán và thông tin |
| **Phân loại** | `/classify` | Phân loại phân khúc xe |
| **Trực quan** | `/visualization` | Biểu đồ và thống kê |

### Demo
**Input mẫu**: Toyota, 2020, 2.0L, Sedan, 50,000km, Trong nước, Tự động, Xăng  
**Output**: ~ 600 triệu VND

**Test sensitivity (ví dụ minh họa, giá trị thay đổi là trung bình toàn bộ dữ liệu):**
- Đổi "Trong nước" → "Nhập khẩu": giá tăng trung bình khoảng +294 triệu VND (tùy từng hãng/dòng xe)
- Đổi "Số sàn" → "Tự động": giá tăng trung bình khoảng +16.7 triệu VND (tùy từng trường hợp)

## Training Models (Tùy chọn)
Models đã được train sẵn. Nếu cần train lại:
```bash
jupyter notebook notebooks/training.ipynb
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



