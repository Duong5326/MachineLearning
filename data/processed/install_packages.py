"""
Script cài đặt các thư viện cần thiết cho huấn luyện mô hình
"""

import sys
import subprocess
import os

def install_packages():
    """Cài đặt các thư viện cần thiết"""
    packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'joblib'
    ]
    
    print("=== CÀI ĐẶT THƯ VIỆN ===")
    for package in packages:
        print(f"Đang cài đặt {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Đã cài đặt {package}")
    
    print("\nĐã cài đặt tất cả thư viện cần thiết!")

if __name__ == "__main__":
    install_packages()