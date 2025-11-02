# Script chạy toàn bộ quy trình thu thập, xử lý và phân tích dữ liệu xe
# Cập nhật: Quy trình hoàn chỉnh từ thu thập dữ liệu đến chuẩn bị cho mô hình máy học

# Kích hoạt môi trường ảo
$env:PYTHONIOENCODING = "utf-8"
Write-Host "===== QUY TRÌNH XỬ LÝ DỮ LIỆU XE BONBANH.COM =====" -ForegroundColor Cyan
Write-Host "`nĐang kích hoạt môi trường ảo Python..." -ForegroundColor Green
& D:\hocmay\.venv\Scripts\Activate.ps1

# Kiểm tra môi trường
Write-Host "`nKiểm tra môi trường Python..." -ForegroundColor Green
python -c "import sys; print(f'Python {sys.version}')"

# Thu thập dữ liệu
Write-Host "`nBước 1: Thu thập dữ liệu xe từ Bonbanh.com..." -ForegroundColor Yellow
$collectOK = $true
try {
    python D:\hocmay\MachineLearning\bonbanh_collector.py
    if ($LASTEXITCODE -ne 0) { $collectOK = $false }
}
catch {
    $collectOK = $false
    Write-Host "Lỗi khi thu thập dữ liệu: $_" -ForegroundColor Red
}

# Xử lý dữ liệu từ raw đến processed
Write-Host "`nBước 2: Xử lý dữ liệu xe (từ raw đến processed)..." -ForegroundColor Yellow
$processOK = $true
try {
    python D:\hocmay\MachineLearning\process_car_data.py
    if ($LASTEXITCODE -ne 0) { $processOK = $false }
}
catch {
    $processOK = $false
    Write-Host "Lỗi khi xử lý dữ liệu: $_" -ForegroundColor Red
}

# Kiểm tra kết quả
Write-Host "`n===== KẾT QUẢ QUY TRÌNH =====" -ForegroundColor Cyan

# Hiển thị số lượng dữ liệu
$trainFile = "D:\hocmay\MachineLearning\data\processed\train_data.csv"
$testFile = "D:\hocmay\MachineLearning\data\processed\test_data.csv"

if (Test-Path $trainFile) {
    $trainCount = (Get-Content $trainFile | Measure-Object -Line).Lines - 1  # Trừ đi dòng header
    Write-Host "Dữ liệu huấn luyện (train): $trainCount mẫu" -ForegroundColor Green
}
else {
    Write-Host "Không tìm thấy file dữ liệu huấn luyện!" -ForegroundColor Red
}

if (Test-Path $testFile) {
    $testCount = (Get-Content $testFile | Measure-Object -Line).Lines - 1  # Trừ đi dòng header
    Write-Host "Dữ liệu kiểm tra (test): $testCount mẫu" -ForegroundColor Green
}
else {
    Write-Host "Không tìm thấy file dữ liệu kiểm tra!" -ForegroundColor Red
}

# Phân tích dữ liệu (nếu có script analyze_car_data.py)
$analyzeScript = "D:\hocmay\MachineLearning\data\analyze_car_data.py"
if (Test-Path $analyzeScript) {
    Write-Host "`nBước 3: Phân tích dữ liệu xe..." -ForegroundColor Yellow
    try {
        python $analyzeScript
    }
    catch {
        Write-Host "Lỗi khi phân tích dữ liệu: $_" -ForegroundColor Red
    }
}

Write-Host "`nQuy trình xử lý dữ liệu đã hoàn tất!" -ForegroundColor Cyan
Write-Host "Các file kết quả nằm trong thư mục: D:\hocmay\MachineLearning\data\processed\" -ForegroundColor Cyan
Write-Host "- Dữ liệu đã làm sạch: car_data.json và processed_car_data.csv"
Write-Host "- Dữ liệu tiếng Anh: car_data_en.json và car_data_en.csv"
Write-Host "- Dữ liệu đã nâng cao: enhanced_car_data.json và enhanced_car_data.csv"
Write-Host "- Tập huấn luyện: train_data.csv"
Write-Host "- Tập kiểm tra: test_data.csv"

# Dừng để đọc thông báo
Write-Host "`nNhấn phím bất kỳ để kết thúc..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")