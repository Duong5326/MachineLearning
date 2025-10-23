# Quy trình thu thập và xử lý dữ liệu xe

## Cấu trúc dữ liệu

### Thư mục `raw/`
Chứa dữ liệu thô từ quá trình thu thập dữ liệu:
- `car_data.json`: Dữ liệu xe thô ở định dạng JSON, chứa trường trùng lặp (cả có dấu và không dấu)
- `car_data.csv`: Dữ liệu xe thô ở định dạng CSV 
- `data/raw/`: Thư mục chứa các file HTML từ Bonbanh.com, dùng để debug:
  - `bonbanh_page1.html`, `bonbanh_page2.html`...: Các trang danh sách xe
  - `bonbanh_detail.html`: Ví dụ trang chi tiết xe
  - `bonbanh_filter.html`: Trang với bộ lọc
- `bonbanh_scraper_raw.py`: Script gốc để thu thập dữ liệu
- `fix_car_data.py`: Script sửa các vấn đề về trường trùng lặp

### Thư mục `processed/`
Chứa dữ liệu đã được xử lý qua nhiều bước:
- **Dữ liệu đã làm sạch (tiếng Việt)**:
  - `car_data.json`: Bản JSON đã loại bỏ trường trùng lặp
  - `processed_car_data.csv`: Bản CSV tương ứng
  
- **Dữ liệu đã làm sạch (tiếng Anh)**:
  - `car_data_en.json`: Bản JSON với tên trường tiếng Anh
  - `car_data_en.csv`: Bản CSV tương ứng

- **Dữ liệu đã nâng cao**:
  - `enhanced_car_data.json`: Dữ liệu đã thêm các trường phân tích (định dạng JSON)
  - `enhanced_car_data.csv`: Bản CSV tương ứng
  
- **Dữ liệu cho mô hình học máy**:
  - `train_data.csv`: Tập huấn luyện (80% dữ liệu)
  - `test_data.csv`: Tập kiểm tra (20% dữ liệu)
  
- **Scripts xử lý dữ liệu**:
  - `clean_raw_to_processed.py`: Làm sạch dữ liệu từ raw sang processed
  - `enhance_car_data.py`: Nâng cao dữ liệu với các trường phân tích thêm
  - `split_train_test.py`: Phân chia dữ liệu thành tập train và test

## Các trường dữ liệu

### Trường cơ bản (từ `car_data_en.csv`)
| Tên trường | Mô tả |
|------------|-------|
| name | Tên đầy đủ của xe |
| brand | Hãng sản xuất |
| price | Giá bán (text, vd: "1 Tỷ 599 Triệu") |
| url | Link đến trang chi tiết xe trên Bonbanh |
| year | Năm sản xuất |
| origin | Xuất xứ (Lắp ráp trong nước/Nhập khẩu) |
| transmission | Loại hộp số |
| body_type | Kiểu dáng xe (Sedan, SUV, Hatchback...) |
| engine | Thông tin động cơ (vd: "Xăng 1.5 L") |
| drive | Hệ dẫn động |
| mileage | Số km đã đi |
| exterior_color | Màu ngoại thất |
| interior_color | Màu nội thất |
| condition | Tình trạng xe (Xe đã dùng/Xe mới) |
| location | Địa chỉ bán xe |

### Trường nâng cao (bổ sung trong `enhanced_car_data.csv`)
| Tên trường | Mô tả |
|------------|-------|
| gia_trieu | Giá bán đã chuyển sang số (đơn vị: triệu đồng) |
| nhien_lieu | Loại nhiên liệu (Xăng/Dầu/Điện/Hybrid) |
| dung_tich | Dung tích động cơ (số lít) |
| tuoi_xe | Tuổi xe (tính từ năm sản xuất) |
| phan_khuc_gia | Phân khúc theo giá (Phổ thông/Trung cấp/Cao cấp/Sang trọng) |
| phan_khuc | Phân khúc theo kiểu dáng và giá (VD: SUV cỡ nhỏ, Sedan hạng B) |

## Quy trình chạy

### Chuẩn bị
1. Kích hoạt môi trường ảo Python
   ```
   D:\hocmay\.venv\Scripts\Activate.ps1
   ```

### Thu thập dữ liệu
2. Chạy script thu thập dữ liệu mới
   ```
   python D:\hocmay\MachineLearning\bonbanh_collector.py
   ```

### Xử lý dữ liệu đầy đủ
3. Chạy quy trình xử lý hoàn chỉnh
   ```
   python D:\hocmay\MachineLearning\process_car_data.py
   ```
   Script này tự động thực hiện tuần tự:
   - Làm sạch dữ liệu (clean_raw_to_processed.py)
   - Nâng cao dữ liệu (enhance_car_data.py)
   - Phân chia thành tập train/test (split_train_test.py)

### Chạy toàn bộ quy trình từ đầu đến cuối
4. Sử dụng script PowerShell tổng hợp
   ```
   powershell -ExecutionPolicy Bypass -File D:\hocmay\MachineLearning\run_car_data_pipeline.ps1
   ```

## Lưu ý
- File **`enhanced_car_data.csv`** là file sạch nhất và phù hợp nhất cho phân tích dữ liệu.
- File **`car_data_en.csv`** là phiên bản cơ bản và gọn nhẹ của dữ liệu nếu không cần các trường phân tích nâng cao.
- Các file **`train_data.csv`** và **`test_data.csv`** đã được chia tách để dùng cho các mô hình học máy.