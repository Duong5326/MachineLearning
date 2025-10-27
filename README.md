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
|--------------------|--------------|----------------|------------|
| Nguyễn Thái Dương  | 23001859     | Duong5326      | Đóng góp 1 |
| Lê Khả Dũng        | 23001847     | github_name    | Đóng góp 2 |
| Nguyễn Hữu Duy     | 23001853     | 23001853-wq    | Đóng góp 3 |


## Cấu trúc Dự án

```
MachineLearning/
├── data/
│   ├── raw/                     # Dữ liệu gốc
│   │   ├── used_cars.csv        # Bộ dữ liệu gốc xe đã qua sử dụng
│   │   ├── bot.py
│   │   ├── bot2.py         
│   ├── processed/               # Dữ liệu sau khi làm sạch
│   │   ├── car_data_en.csv      # Dữ liệu đã chuyển sang tiếng Anh
│   │   ├── car_data_en.json     # Dữ liệu JSON đã chuyển sang tiếng Anh
│   │   ├── enhanced_car_data.csv # Dữ liệu đã tăng cường
│   │   ├── enhanced_car_data.json # Dữ liệu JSON đã tăng cường
│   │   ├── processed_car_data.csv # Dữ liệu đã xử lý
│   │   ├── train_data.csv       # Dữ liệu huấn luyện
│   │   ├── test_data.csv        # Dữ liệu kiểm tra
│   │   ├── clean_raw_to_processed.py # Script làm sạch dữ liệu
│   │   ├── enhance_car_data.py  # Script tăng cường đặc trưng
│   │   ├── split_train_test.py  # Script chia tập train/test
│   │   ├── run_train_model.bat  # Batch file chạy quá trình huấn luyện
│   │   ├── install_packages.py  # Script cài đặt thư viện cần thiết
│   │   ├── models/              # Thư mục lưu mô hình đã huấn luyện 
│   │   └── processed_analysis/  # Thư mục lưu dữ liệu phân tích, scaler, PCA, các file .pkl trung gian
│   └── README.md                # Mô tả nguồn dữ liệu và cấu trúc
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Phân tích khám phá dữ liệu
│   ├── 02_data_preprocessing.ipynb  # Tiền xử lý dữ liệu
│   ├── 03_regression_models.ipynb   # Mô hình hồi quy dự đoán giá
│   ├── 04_clustering_analysis.ipynb # Phân cụm các loại ô tô
│   ├── 05_data_dimensionality_reduction.ipynb # Phân tích phân cụm
│   ├── 06_model_analysis.ipynb      # Phân tích chi tiết mô hình
│   ├── 07_classification_models.ipynb # Mô hình phân loại phân khúc giá
│   └── car_price_model_analysis_simple.ipynb         # Phân tích kết quả (tóm tắt)
├── src/
│   ├── data_loader.py           # Đọc và xử lý dữ liệu
│   ├── preprocessing.py         # Tiền xử lý dữ liệu
│   ├── feature_engineering.py   # Tạo đặc trưng mới
│   ├── dimensionality_reduction.py         # Giam chiều trực quan hóa dữ liệu
│   ├── visualization.py         # Các hàm vẽ biểu đồ
│   ├── regression.py            # Các mô hình hồi quy
│   ├── classification.py        # Các mô hình phân loại
│   ├── clustering.py            # Các mô hình phân cụm
│   ├── evaluate.py              # Đánh giá mô hình
│   ├── predict.py               # Dự đoán giá xe mới
│   ├── train_price_model.py     # Script huấn luyện mô hình dự đoán giá

├── static/
│   └── css/                     # CSS cho giao diện web
│       └── style.css
├── templates/
│   ├── index.html               # Trang chủ
│   ├── predict.html             # Trang dự đoán giá
│   ├── classify.html            # Trang phân loại phân khúc
│   └── visualization.html       # Trang hiển thị biểu đồ trực quan
├── application.py               # Ứng dụng Flask
├── predict_car_price.py         # Script dự đoán giá xe
├── process_car_data.py          # Script xử lý dữ liệu xe
├── run_car_data_pipeline.ps1    # PowerShell script chạy pipeline xử lý dữ liệu
├── requirements.txt             # Danh sách các thư viện cần thiết
├── README.md                    # Mô tả dự án, hướng dẫn cài đặt và sử dụng
```

## Tổng quan dự án

Dự án này tập trung vào việc phân tích, dự đoán và phân loại giá ô tô cũ tại Việt Nam. Mục tiêu chính bao gồm:

1. **Phân tích Khám phá Dữ liệu**: Tìm hiểu mối quan hệ giữa các đặc tính của xe (như thương hiệu, năm sản xuất, số km, v.v.) và giá cả.
2. **Kỹ thuật Đặc trưng**: Tạo các đặc trưng có giá trị như tuổi xe, phân loại thương hiệu, tỷ lệ số km/năm.
3. **Dự đoán Giá Xe**: Phát triển nhiều mô hình hồi quy để dự đoán giá xe chính xác.
4. **Phân loại Phân Khúc Giá**: Sử dụng mô hình Naive Bayes, Random Forest, và SVM để phân loại xe vào các phân khúc giá khác nhau.
5. **Phân Cụm Thị Trường**: Phân cụm các loại xe dựa trên đặc tính kỹ thuật và giá trị.
6. **Phân Tích Mô Hình**: Đánh giá hiệu suất các mô hình dự đoán và phân loại chi tiết.
7. **Ứng Dụng Web**: Phát triển giao diện web cho phép người dùng tương tác với các mô hình.

## Dữ liệu

Bộ dữ liệu chứa thông tin về xe ô tô đã qua sử dụng với các thông tin:
- Đường dẫn dữ liệu (url)
- Hãng xe (brand)
- Năm sản xuất (year)
- Số km đã đi (mileage)
- Xuất xứ (origin)
- Kiểu dáng (body_type)
- Hộp số (transmission)
- Động cơ (engine)
- Màu ngoại thất (exterior color)
- Màu nội thất (interior color)
- Số chỗ ngồi (seats)
- Số cửa (doors)
- dẫn động (drive)
- Giá bán (price)
- Tình trạng xe (condition)

## Cài đặt

1. Clone repository:
   ```
   git clone <url-repo>
   cd MachineLearnings
   ```

2. Cài đặt các thư viện cần thiết:
   ```
   pip install -r requirements.txt
   ```

3. Chạy quy trình xử lý dữ liệu và huấn luyện mô hình:
   ```
   powershell -ExecutionPolicy Bypass -File run_car_data_pipeline.ps1
   ```
   (Các model sẽ được lưu vào data/processed/models/)

4. Chạy ứng dụng web:
   ```
   python application.py
   ```

## Sử dụng

### Quy trình xử lý dữ liệu

Để xử lý dữ liệu và huấn luyện mô hình, sử dụng script PowerShell:

```powershell
.\run_car_data_pipeline.ps1
```

Script này sẽ thực hiện các bước:
1. Làm sạch dữ liệu thô
2. Tăng cường các đặc trưng
3. Chia dữ liệu thành tập huấn luyện và kiểm thử
4. Huấn luyện các mô hình

### Dự đoán giá xe

Để dự đoán giá xe với dữ liệu mới, sử dụng:

```bash
python predict_car_price.py --brand "Toyota" --year 2018 --mileage 50000 --body_type "Sedan"
```

**Lưu ý:**
- Đường dẫn model trong các script đã được chuẩn hóa về `data/processed/models/` (ví dụ: `Lasso_model.pkl`, `RandomForest_model.pkl`...)
- Nếu gặp lỗi không tìm thấy file model, hãy kiểm tra lại vị trí file `.pkl` trong thư mục này.

### Xử lý dữ liệu

Để chỉ xử lý dữ liệu mà không huấn luyện mô hình:

```bash
python process_car_data.py --input data/raw/used_cars.csv --output data/processed/processed_car_data.csv
```

### Jupyter Notebooks

Các notebooks được tổ chức theo thứ tự phân tích:

1. `01_data_exploration.ipynb`: Khám phá và phân tích dữ liệu ban đầu
2. `02_data_preprocessing.ipynb`: Làm sạch và tiền xử lý dữ liệu
3. `03_regression_models.ipynb`: Phát triển mô hình hồi quy dự đoán giá xe
4. `04_clustering_analysis.ipynb`: Phân tích phân cụm các loại xe
5. `05_data_dimensionality_reduction.ipynb`: Giảm chiều dữ liệu, lưu các preprocessing model (PCA, scaler, ...) vào `data/processed/processed_analysis/`
6. `06_model_analysis.ipynb`: Phân tích chi tiết hiệu suất mô hình
7. `07_classification_models.ipynb`: Phát triển mô hình phân loại phân khúc giá
8. `car_price_model_analysis_simple.ipynb`: Phân tích kết quả (tóm tắt)

### Ứng dụng Web

Sau khi khởi chạy ứng dụng, truy cập http://localhost:5000 để sử dụng các tính năng:

- **Trang chủ**: Giới thiệu tổng quan về ứng dụng
- **Dự đoán giá**: Nhập các thông số của xe để dự đoán giá
- **Trực quan hóa**: Hiển thị các biểu đồ phân tích dữ liệu

### Huấn luyện mô hình

Có nhiều cách để huấn luyện mô hình:

1. **Sử dụng run_car_data_pipeline.ps1** (khuyến nghị):
   ```
   powershell -ExecutionPolicy Bypass -File run_car_data_pipeline.ps1
   ```
   Script này điều phối toàn bộ quy trình từ xử lý dữ liệu đến huấn luyện và đánh giá.

2. **Sử dụng train_price_model.py**:
   ```
   python src/train_price_model.py
   ```
   Script này tập trung vào việc huấn luyện các mô hình và lưu vào thư mục `data/processed/models`.

   **Lưu ý:**
   - Các preprocessing model (PCA, scaler, ...) và dữ liệu trung gian sẽ được lưu vào `data/processed/processed_analysis/`.

3. **Sử dụng Jupyter Notebook**:
   Các notebook trong thư mục `notebooks/` cung cấp phương pháp tiếp cận tương tác với các phân tích trực quan.

## Các mô hình được triển khai

### Mô hình Hồi quy (Dự đoán giá chính xác)
1. Linear Regression (Hồi quy tuyến tính)
2. Ridge, Lasso Regression
3. Random Forest Regressor
4. Gradient Boosting

### Mô hình Phân loại (Phân khúc giá)
1. Naive Bayes (GaussianNB)
2. Random Forest Classifier 
3. Support Vector Machine (SVM)

### Mô hình Phân cụm
1. K-Means Clustering
2. Hierarchical Clustering
3. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

## Kỹ thuật đặc trưng

Các đặc trưng được tạo ra từ dữ liệu gốc:
- `car_age`: Tuổi của xe (dựa trên năm sản xuất)
- `price_million`: Giá xe (đơn vị: triệu đồng)
- `mileage_km`: Số km đã đi (chuẩn hóa)
- `mileage_per_year`: Số km đi trung bình mỗi năm
- `brand_category`: Phân loại thương hiệu (Luxury, Premium, Economy)
- `processed_analysis/`: Lưu các preprocessing model (PCA, scaler, encoder, ...) và dữ liệu phân tích trung gian

## Đánh giá hiệu suất

### Mô hình Hồi quy
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score (Coefficient of determination)
- Adjusted R² Score (Adjusted Coefficient of determination)

### Mô hình Phân loại
- Accuracy (Độ chính xác)
- Precision (Độ chính xác dương tính)
- Recall (Độ nhạy)
- F1 Score (Điểm F1)
- Confusion Matrix (Ma trận nhầm lẫn)
- ROC Curve & AUC (Đường cong ROC & Diện tích dưới đường cong)

## Thư viện sử dụng

- **Xử lý dữ liệu**: pandas, numpy
- **Học máy**: scikit-learn, TensorFlow/Keras
- **Trực quan hóa**: matplotlib, seaborn, plotly
- **Web App**: Flask, HTML, CSS
- **Tối ưu hóa mô hình**: GridSearchCV, RandomizedSearchCV
- **Tiền xử lý**: SimpleImputer, StandardScaler, OneHotEncoder, LabelEncoder
- **Đánh giá mô hình**: classification_report, confusion_matrix, cross_val_score

