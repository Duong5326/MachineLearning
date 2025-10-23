# Process used_cars.csv
# Script PowerShell để chạy quá trình làm sạch và nâng cao dữ liệu xe từ used_cars.csv

# Đường dẫn đến thư mục hiện tại
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$processedDir = Join-Path -Path $scriptPath -ChildPath "data\processed"

# Kiểm tra thư mục processed tồn tại
if (-not (Test-Path $processedDir)) {
    Write-Host "Tạo thư mục processed..."
    New-Item -Path $processedDir -ItemType Directory -Force | Out-Null
}

# 1. Chạy script làm sạch dữ liệu
Write-Host "1. Đang làm sạch dữ liệu từ used_cars.csv..." -ForegroundColor Green
$cleanScript = Join-Path -Path $processedDir -ChildPath "clean_raw_to_processed.py"
python $cleanScript

# 2. Chạy script nâng cao dữ liệu
Write-Host "`n2. Đang nâng cao dữ liệu..." -ForegroundColor Green
$enhanceScript = Join-Path -Path $processedDir -ChildPath "enhance_car_data.py"
python $enhanceScript

# 3. Hiển thị kết quả
Write-Host "`n3. Hoàn tất xử lý dữ liệu!" -ForegroundColor Green

$enhancedCsv = Join-Path -Path $processedDir -ChildPath "enhanced_car_data.csv"
if (Test-Path $enhancedCsv) {
    $rowCount = (Import-Csv $enhancedCsv | Measure-Object).Count
    Write-Host "`nDữ liệu đã được làm sạch và nâng cao với $rowCount dòng."
    Write-Host "File dữ liệu đã xử lý: $enhancedCsv"
} else {
    Write-Host "`nKhông tìm thấy file enhanced_car_data.csv. Vui lòng kiểm tra lỗi trong quá trình xử lý." -ForegroundColor Red
}

Write-Host "`nNhấn phím bất kỳ để thoát..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")