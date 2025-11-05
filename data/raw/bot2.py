import requests
from bs4 import BeautifulSoup
import csv
import re
import time
from tqdm import tqdm

BASE_URL = "https://bonbanh.com"
LIST_URL = f"{BASE_URL}/ha-noi/oto-cu-da-qua-su-dung"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

# ====== HÀM CHUYỂN ĐỔI GIÁ ======
def parse_price(price_str):
    """Chuyển '1 Tỷ 10 Triệu' -> 1010000000"""
    price_str = price_str.lower().strip()
    ty = re.search(r"(\d+)\s*t[ỷi]", price_str)
    trieu = re.search(r"(\d+)\s*triệu", price_str)

    total = 0
    if ty:
        total += int(ty.group(1)) * 1_000_000_000
    if trieu:
        total += int(trieu.group(1)) * 1_000_000
    return total if total > 0 else None

# ====== HÀM LẤY SỐ TRANG CUỐI ======
def get_max_page():
    try:
        resp = requests.get(LIST_URL, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        pages = soup.select("div.navpage span.bbl")
        page_nums = []
        for sp in pages:
            if "url" in sp.attrs and "page" in sp["url"]:
                num = sp["url"].split(",")[-1]
                if num.isdigit():
                    page_nums.append(int(num))
        if page_nums:
            return max(page_nums)
    except Exception as e:
        print("Không thể xác định số trang, dùng mặc định 1:", e)
    return 1

# ====== HÀM XỬ LÝ TÊN XE ======
def clean_car_name(name):
    name = name.strip()
    # bỏ chữ "Xe" ở đầu
    name = re.sub(r"^Xe\s+", "", name, flags=re.IGNORECASE)
    # bỏ phần năm phía trước dấu gạch
    name = re.sub(r"\s*-\s*\d{4}.*$", "", name)
    # bỏ phần số động cơ, hộp số, dẫn động trong tên
    name = re.sub(r"\b\d\.\d+[LT]?\b", "", name)  # động cơ 1.5L
    name = re.sub(r"\bAT\b|\bMT\b", "", name)     # hộp số
    name = re.sub(r"\b4x4\b|\bAWD\b|\bFWD\b|\bRWD\b", "", name)  # dẫn động
    name = re.sub(r"\s{2,}", " ", name)  # xóa dư khoảng trắng
    return name.strip()

# ====== HÀM LẤY DỮ LIỆU XE CHI TIẾT ======
def parse_car_detail(url):
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        box = soup.select_one("div.box_car_detail")
        if not box:
            return None

        data = {}

        for row in box.select("div.row, div.row_last"):
            label = row.select_one(".label label")
            value = row.select_one(".txt_input .inp")
            if not label or not value:
                continue

            key = label.get_text(strip=True)
            val = value.get_text(strip=True)

            if key == "Năm sản xuất:":
                data["Năm"] = val
            elif key == "Số Km đã đi:":
                data["Số km"] = val
            elif key == "Xuất xứ:":
                data["Xuất xứ"] = val
            elif key == "Kiểu dáng:":
                data["Kiểu dáng"] = val
            elif key == "Hộp số:":
                data["Hộp số"] = val
            elif key == "Động cơ:":
                data["Động cơ"] = val
            elif key == "Màu ngoại thất:":
                data["Màu ngoại thất"] = val
            elif key == "Màu nội thất:":
                data["Màu nội thất"] = val
            elif key == "Số chỗ ngồi:":
                data["Số chỗ"] = val
            elif key == "Dẫn động:":
                data["Dẫn động"] = val

        return data
    except Exception as e:
        print(f"Lỗi khi tải chi tiết {url}: {e}")
        return None

# ====== CHƯƠNG TRÌNH CHÍNH ======
def main():
    max_page = get_max_page()
    print(f"Tổng số trang phát hiện: {max_page}")

    csv_file = open("bonbanh_data.csv", "w", newline="", encoding="utf-8-sig")
    writer = csv.writer(csv_file)
    writer.writerow([
        "Tên xe", "Năm", "Số km", "Xuất xứ", "Kiểu dáng",
        "Hộp số", "Động cơ", "Màu ngoại thất", "Màu nội thất",
        "Số chỗ", "Dẫn động", "Giá (chuỗi)", "Giá (VNĐ)", "URL"
    ])

    for page in range(1, max_page + 1):
        url = f"{LIST_URL}/page,{page}"
        print(f"==> Đang xử lý trang {page}: {url}")
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(resp.text, "html.parser")
            car_items = soup.select('a[itemprop="url"]')

            if not car_items:
                print(f"Không tìm thấy xe nào trong trang {page}.")
                continue

            for a in tqdm(car_items, desc=f"Trang {page} - {len(car_items)} xe"):
                href = a.get("href")
                full_url = BASE_URL + "/" + href.lstrip("/")
                title = a.get("title", "")
                clean_name = clean_car_name(title)
                # lấy phần sau dấu "-" để lấy giá
                price_str = title.split("-")[-1].strip() if "-" in title else "N/A"
                price_num = parse_price(price_str)

                detail = parse_car_detail(full_url) or {}
                writer.writerow([
                    clean_name,
                    detail.get("Năm", ""),
                    detail.get("Số km", ""),
                    detail.get("Xuất xứ", ""),
                    detail.get("Kiểu dáng", ""),
                    detail.get("Hộp số", ""),
                    detail.get("Động cơ", ""),
                    detail.get("Màu ngoại thất", ""),
                    detail.get("Màu nội thất", ""),
                    detail.get("Số chỗ", ""),
                    detail.get("Dẫn động", ""),
                    price_str,
                    price_num if price_num else "",
                    full_url
                ])

            time.sleep(2)

        except Exception as e:
            print(f"Lỗi khi tải trang {url}: {e}")
            time.sleep(10)
            continue

    csv_file.close()
    print("✅ Hoàn tất! Dữ liệu được lưu vào 'bonbanh_data.csv'")

if __name__ == "__main__":
    main()
