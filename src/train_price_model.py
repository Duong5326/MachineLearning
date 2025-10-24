"""
Script huấn luyện mô hình dự đoán giá xe dựa trên các đặc điểm
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import re
import joblib

# Đường dẫn file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
TRAIN_FILE = os.path.join(PROCESSED_DIR, "train_data.csv")
TEST_FILE = os.path.join(PROCESSED_DIR, "test_data.csv")
# Lưu model vào data/processed/models
MODEL_DIR = os.path.join(PROCESSED_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

def convert_price_to_number(price_text):
    """Chuyển đổi giá từ dạng văn bản (vd: 1 Tỷ 599 Triệu) sang số (đơn vị: triệu)."""
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

def prepare_data():
    """Đọc và chuẩn bị dữ liệu từ file CSV"""
    print("Đọc dữ liệu từ", TRAIN_FILE)
    train_df = pd.read_csv(TRAIN_FILE)
    
    print("Đọc dữ liệu từ", TEST_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    # Chuyển đổi giá thành số
    train_df['price_million'] = train_df['price'].apply(convert_price_to_number)
    test_df['price_million'] = test_df['price'].apply(convert_price_to_number)
    
    # Chuyển đổi năm thành số nguyên
    train_df['year'] = pd.to_numeric(train_df['year'], errors='coerce')
    test_df['year'] = pd.to_numeric(test_df['year'], errors='coerce')
    
    # Điền các giá trị bị thiếu với giá trị trung bình hoặc phổ biến nhất
    print("Xử lý các giá trị bị thiếu (NaN)...")
    # Điền giá trị thiếu cho các cột số (không dùng inplace để tránh FutureWarning)
    numeric_cols = ['year', 'price_million']
    for col in numeric_cols:
        train_df[col] = train_df[col].fillna(train_df[col].median())
        test_df[col] = test_df[col].fillna(test_df[col].median())
    
    # Trích xuất thông tin nhiên liệu và dung tích từ trường engine
    def extract_fuel_type(engine_text):
        if pd.isna(engine_text) or not engine_text:
            return "Unknown"
        if "Xăng" in engine_text:
            return "Gasoline"
        elif "Dầu" in engine_text:
            return "Diesel"
        elif "Điện" in engine_text:
            return "Electric"
        elif "Hybrid" in engine_text:
            return "Hybrid"
        return "Other"
    
    def extract_engine_capacity(engine_text):
        if pd.isna(engine_text) or not engine_text:
            return np.nan
        capacity_match = re.search(r'(\d+\.\d+|\d+)\s*L', str(engine_text))
        if capacity_match:
            return float(capacity_match.group(1))
        return np.nan
    
    # Thêm các đặc trưng mới
    train_df['fuel_type'] = train_df['engine'].apply(extract_fuel_type)
    train_df['engine_capacity'] = train_df['engine'].apply(extract_engine_capacity)
    test_df['fuel_type'] = test_df['engine'].apply(extract_fuel_type)
    test_df['engine_capacity'] = test_df['engine'].apply(extract_engine_capacity)
    
    # Thêm đặc trưng tuổi xe
    current_year = 2025  # Đặt năm hiện tại
    train_df['car_age'] = current_year - train_df['year']
    test_df['car_age'] = current_year - test_df['year']
    
    # Thêm đặc trưng xuất xứ (nhập khẩu/lắp ráp)
    train_df['is_imported'] = train_df['origin'].apply(lambda x: 1 if 'Nhập khẩu' in str(x) else 0)
    test_df['is_imported'] = test_df['origin'].apply(lambda x: 1 if 'Nhập khẩu' in str(x) else 0)
    
    # Nếu có cột mileage thì xử lý, nếu không thì bỏ qua
    if 'mileage' in train_df.columns and 'mileage' in test_df.columns:
        def convert_mileage(mileage_text):
            if pd.isna(mileage_text) or not mileage_text:
                return np.nan
            if isinstance(mileage_text, (int, float)):
                return mileage_text
            mileage_str = re.sub(r'[^0-9]', '', str(mileage_text))
            return float(mileage_str) if mileage_str else np.nan
        train_df['mileage_km'] = train_df['mileage'].apply(convert_mileage)
        test_df['mileage_km'] = test_df['mileage'].apply(convert_mileage)
        train_df['mileage_km'] = train_df['mileage_km'].fillna(train_df['mileage_km'].median())
        test_df['mileage_km'] = test_df['mileage_km'].fillna(test_df['mileage_km'].median())
    elif 'mileage_km' in train_df.columns and 'mileage_km' in test_df.columns:
        # Nếu đã có sẵn mileage_km thì chỉ cần điền giá trị thiếu
        train_df['mileage_km'] = train_df['mileage_km'].fillna(train_df['mileage_km'].median())
        test_df['mileage_km'] = test_df['mileage_km'].fillna(test_df['mileage_km'].median())
    else:
        # Không có cột mileage hoặc mileage_km, bỏ qua đặc trưng này
        print("Không có cột mileage hoặc mileage_km trong dữ liệu, sẽ không dùng đặc trưng quãng đường đã đi.")

    # Loại bỏ các dòng có mileage_km = 0 hoặc NaN (nếu có cột này) trước khi binning
    if 'mileage_km' in train_df.columns:
        before = len(train_df)
        train_df = train_df[(train_df['mileage_km'].notna()) & (train_df['mileage_km'] > 0)]
        print(f"Đã loại bỏ {before - len(train_df)} dòng train có mileage_km = 0 hoặc NaN.")
    if 'mileage_km' in test_df.columns:
        before = len(test_df)
        test_df = test_df[(test_df['mileage_km'].notna()) & (test_df['mileage_km'] > 0)]
        print(f"Đã loại bỏ {before - len(test_df)} dòng test có mileage_km = 0 hoặc NaN.")

    # Thêm đặc trưng phân loại số km đã đi (binning) — thực hiện sau khi đã loại bỏ các giá trị không hợp lệ
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
    if 'mileage_km' in train_df.columns:
        train_df['mileage_km'] = train_df['mileage_km'].apply(mileage_bin)
    if 'mileage_km' in test_df.columns:
        test_df['mileage_km'] = test_df['mileage_km'].apply(mileage_bin)
    
    # Điền giá trị thiếu cho cột engine_capacity (không dùng inplace để tránh FutureWarning)
    train_df['engine_capacity'] = train_df['engine_capacity'].fillna(train_df['engine_capacity'].median())
    test_df['engine_capacity'] = test_df['engine_capacity'].fillna(test_df['engine_capacity'].median())



    print("\nTập huấn luyện:")
    print(f"- Số lượng mẫu: {len(train_df)}")
    print(f"- Các đặc trưng: {train_df.columns.tolist()}")

    return train_df, test_df

def train_and_evaluate_models(train_df, test_df):
    """Huấn luyện và đánh giá các mô hình"""
    
    # Chọn các đặc trưng và biến mục tiêu
    feature_cols = ['year', 'engine_capacity', 'car_age', 'is_imported']
    categorical_cols = ['brand', 'body_type', 'fuel_type']
    # Nếu có cột mileage_km (sau khi đã binning), chỉ thêm vào categorical_cols
    if 'mileage_km' in train_df.columns and 'mileage_km' in test_df.columns:
        categorical_cols.append('mileage_km')
        # Đảm bảo mileage_km không nằm trong feature_cols nếu có
        if 'mileage_km' in feature_cols:
            feature_cols.remove('mileage_km')
    target_col = 'price_million'
    
    # Kiểm tra xem các cột có tồn tại không
    for col in feature_cols + categorical_cols:
        if col not in train_df.columns:
            print(f"Cảnh báo: Cột {col} không có trong dữ liệu. Bỏ qua.")
            if col in feature_cols:
                feature_cols.remove(col)
            if col in categorical_cols:
                categorical_cols.remove(col)
    
    # Tạo pipeline tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), feature_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Định nghĩa các mô hình cần huấn luyện
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # Các kết quả đánh giá mô hình
    results = {}
    feature_importance = {}
    
    # Chuẩn bị dữ liệu
    X_train = train_df[feature_cols + categorical_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols + categorical_cols]
    y_test = test_df[target_col]
    
    # Huấn luyện và đánh giá từng mô hình
    for name, model in models.items():
        print(f"\nĐang huấn luyện mô hình {name}...")
        # Tạo pipeline kết hợp tiền xử lý và mô hình
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        # Huấn luyện mô hình
        pipeline.fit(X_train, y_train)
        # Dự đoán trên tập kiểm tra
        y_pred = pipeline.predict(X_test)
        # Tính toán các chỉ số đánh giá
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # Lưu kết quả
        results[name] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2
        }
        # Lưu mô hình
        model_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Đã lưu mô hình {name} vào {model_path}")
        # Nếu là Lasso thì lưu thêm với tên Lasso_model.pkl để Flask sử dụng đúng pipeline
        if name == 'Lasso':
            lasso_pipeline_path = os.path.join(MODEL_DIR, "Lasso_model.pkl")
            joblib.dump(pipeline, lasso_pipeline_path)
            print(f"Đã lưu riêng pipeline Lasso vào {lasso_pipeline_path}")
        # Trích xuất tầm quan trọng của đặc trưng (chỉ cho RandomForest và GradientBoosting)
        if hasattr(model, 'feature_importances_'):
            # Tạo tên các đặc trưng sau khi mã hóa
            feature_names = feature_cols.copy()
            # Thêm các đặc trưng mã hóa one-hot
            cat_encoder = pipeline.named_steps['preprocessor'].transformers_[1][1]
            if hasattr(cat_encoder, 'get_feature_names_out'):
                cat_features = cat_encoder.get_feature_names_out(categorical_cols)
                feature_names.extend(cat_features)
            # Lấy tầm quan trọng của đặc trưng từ mô hình
            feature_importance[name] = dict(zip(
                feature_names, 
                model.feature_importances_
            ))
    
    # In kết quả
    print("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
    for name, metrics in results.items():
        print(f"\nMô hình: {name}")
        for metric_name, value in metrics.items():
            print(f"- {metric_name}: {value:.4f}")
    
    # Xác định mô hình tốt nhất dựa trên R2 Score
    best_model = max(results.items(), key=lambda x: x[1]['R2'])
    print(f"\n=== MÔ HÌNH TỐT NHẤT: {best_model[0]} ===")
    print(f"R2 Score: {best_model[1]['R2']:.4f}")
    print(f"RMSE: {best_model[1]['RMSE']:.4f} triệu đồng")
    
    # Trực quan hóa kết quả
    plt.figure(figsize=(10, 6))
    
    # So sánh R2 Score giữa các mô hình
    r2_scores = {name: metrics['R2'] for name, metrics in results.items()}
    plt.subplot(1, 2, 1)
    plt.bar(r2_scores.keys(), r2_scores.values())
    plt.title('R2 Score by Model')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # So sánh RMSE giữa các mô hình
    rmse_scores = {name: metrics['RMSE'] for name, metrics in results.items()}
    plt.subplot(1, 2, 2)
    plt.bar(rmse_scores.keys(), rmse_scores.values())
    plt.title('RMSE by Model')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'model_comparison.png'))
    print(f"Đã lưu biểu đồ so sánh mô hình vào {os.path.join(MODEL_DIR, 'model_comparison.png')}")
    
    # Trực quan hóa dự đoán vs giá trị thực
    best_model_name = best_model[0]
    best_pipeline = joblib.load(os.path.join(MODEL_DIR, f"{best_model_name}_model.pkl"))
    y_pred_best = best_pipeline.predict(X_test)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred_best, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.title(f'Predicted vs Actual Price ({best_model_name})')
    plt.xlabel('Actual Price (million VND)')
    plt.ylabel('Predicted Price (million VND)')
    plt.savefig(os.path.join(MODEL_DIR, 'prediction_vs_actual.png'))
    print(f"Đã lưu biểu đồ dự đoán vs thực tế vào {os.path.join(MODEL_DIR, 'prediction_vs_actual.png')}")
    
    # Trực quan hóa tầm quan trọng của đặc trưng cho mô hình tốt nhất
    if best_model_name in feature_importance:
        importances = feature_importance[best_model_name]
        # Sắp xếp theo tầm quan trọng giảm dần
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
        
        plt.figure(figsize=(10, 6))
        plt.bar(sorted_importances.keys(), sorted_importances.values())
        plt.title(f'Feature Importances ({best_model_name})')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'))
        print(f"Đã lưu biểu đồ tầm quan trọng đặc trưng vào {os.path.join(MODEL_DIR, 'feature_importance.png')}")
    
    return results, best_model_name

def main():
    """Hàm chính chạy quy trình huấn luyện và đánh giá mô hình"""
    print("=== HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN GIÁ XE ===")
    
    # Chuẩn bị dữ liệu
    train_df, test_df = prepare_data()
    
    # Huấn luyện và đánh giá mô hình
    results, best_model = train_and_evaluate_models(train_df, test_df)
    
    print("\n=== QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT ===")
    print(f"Mô hình tốt nhất: {best_model}")
    print(f"Các mô hình đã được lưu vào thư mục: {MODEL_DIR}")

if __name__ == "__main__":
    main()