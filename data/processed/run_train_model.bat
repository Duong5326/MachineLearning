@echo off
echo ===== HUẤN LUYỆN MÔ HÌNH DỰ ĐOÁN GIÁ XE =====
echo.

echo 1. Kích hoạt môi trường Python
call D:\hocmay\.venv\Scripts\activate.bat

echo.
echo 2. Cài đặt các thư viện cần thiết
python D:\hocmay\MachineLearning\data\processed\install_packages.py

echo.
echo 3. Huấn luyện mô hình
python D:\hocmay\MachineLearning\data\processed\train_price_model.py

echo.
echo ===== QUÁ TRÌNH HOÀN TẤT =====
echo Các mô hình đã được lưu vào thư mục: %~dp0\models\
echo.

pause