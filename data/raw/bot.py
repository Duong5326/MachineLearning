#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated bot.py for bonbanh.com
- Default: allow duplicates (no filtering).
- Better price parsing: ignore year-like numbers, prefer groups with units.
- Better name cleaning: remove engine tokens (1.6T, 2.0L), AT/MT, drive tokens; keep model digits like 'Mazda 3'.
- If Dong_co/Dan_dong missing in detail, try to extract from title.
"""
from __future__ import annotations
import re, csv, json, os, time, random, argparse
from typing import Optional, Dict, List, Tuple, Set
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kw): return x

# CONFIG
START_URL = "https://bonbanh.com/ha-noi/oto-cu-da-qua-su-dung"
PAGE_URL_PATTERN = "https://bonbanh.com/ha-noi/oto-cu-da-qua-su-dung/page,{}"
OUTPUT_CSV = "used_cars.csv"
PROGRESS_FILE = "progress.json"

USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8"}

MIN_DELAY, MAX_DELAY = 1.5, 3.8
PAGE_DELAY = (2.0, 5.0)
SAFE_GET_RETRIES = 4
SAFE_GET_BACKOFF = 2.0

FIELDNAMES = [
    "Ten_xe", "Nam_san_xuat", "So_km", "Xuat_xu", "Kieu_dang", "Hop_so",
    "Dong_co", "Mau_ngoai_that", "Mau_noi_that", "So_cho_ngoi", "Dan_dong",
    "Gia_ban_text", "Gia_ban", "Link"
]

LABEL_MAP = {
    "Năm sản xuất": "Nam_san_xuat",
    "Số Km đã đi": "So_km",
    "Số Km": "So_km",
    "Xuất xứ": "Xuat_xu",
    "Kiểu dáng": "Kieu_dang",
    "Hộp số": "Hop_so",
    "Động cơ": "Dong_co",
    "Màu ngoại thất": "Mau_ngoai_that",
    "Màu nội thất": "Mau_noi_that",
    "Số chỗ ngồi": "So_cho_ngoi",
    "Số chỗ": "So_cho_ngoi",
    "Dẫn động": "Dan_dong",
}

# Session + retry
session = requests.Session()
retry_strategy = Retry(total=6, status_forcelist=[429,500,502,503,504],
                       allowed_methods=["HEAD","GET","OPTIONS"], backoff_factor=1)
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
session.mount("http://", HTTPAdapter(max_retries=retry_strategy))
session.headers.update(HEADERS)

# Helpers
def save_progress(progress: Dict) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)

def load_progress() -> Dict:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def safe_get(url: str, timeout: float = 15.0) -> Optional[requests.Response]:
    for attempt in range(1, SAFE_GET_RETRIES + 1):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            wait = SAFE_GET_BACKOFF * attempt + random.uniform(0,1.5)
            print(f"[safe_get] lỗi khi tải {url}: {e} — thử sau {wait:.1f}s (lần {attempt})")
            time.sleep(wait)
    print(f"[safe_get] bỏ qua {url} sau {SAFE_GET_RETRIES} lần.")
    return None

def normalize_text(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def parse_price_text_to_int(price_text: Optional[str]) -> Optional[int]:
    """Robust price parsing:
    - Prefer groups that have units (tỷ/triệu/nghìn)
    - Ignore year-like numbers (1900-2099)
    - If only one number and no unit, heuristics: if >=1e6 treat as VND; if <10000 treat as millions
    """
    if not price_text:
        return None
    text = price_text.lower().strip()
    if any(k in text for k in ("thỏa", "thoả", "liên hệ", "lien he", "thoa thuan")):
        return None
    text = text.replace(",", ".")
    # find all (num, unit) groups
    groups = re.findall(r"([\d\.]+)\s*(tỷ|ty|triệu|trieu|nghìn|nghin|k|vnđ|vnd)?", text)
    # filter out empty matches
    groups = [(n, u or "") for (n,u) in groups if n.strip()]
    # remove any groups where n is a year (1900-2099) — avoid year bleeding into price
    filtered = []
    for n,u in groups:
        try:
            if '.' in n:
                v = float(n)
            else:
                v = int(n)
            if 1900 <= int(float(n)) <= 2099:
                # skip as it's a year
                continue
        except:
            pass
        filtered.append((n,u))
    groups = filtered
    # if any group has a unit -> compute using those (ignore groups with no unit)
    unit_groups = [ (n,u) for (n,u) in groups if u ]
    if unit_groups:
        total = 0
        for n,u in unit_groups:
            try:
                val = float(n)
            except:
                continue
            u = u.strip()
            if u in ("tỷ","ty"):
                total += int(val * 1_000_000_000)
            elif u in ("triệu","trieu"):
                total += int(val * 1_000_000)
            elif u in ("nghìn","nghin","k"):
                total += int(val * 1_000)
            elif u in ("vnđ","vnd"):
                total += int(val)
            else:
                total += int(val)
        return int(total) if total else None
    # else no units found, but maybe single number remains (not year)
    nums = [n for (n,u) in groups]
    if not nums:
        # fallback: try capture big digits
        m = re.search(r"(\d{6,})", text)
        if m:
            try:
                return int(m.group(1))
            except:
                return None
        return None
    # heuristics:
    if len(nums) == 1:
        try:
            val = float(nums[0])
            if val >= 1_000_000:
                return int(val)
            if val >= 1000 and val < 1_000_000:
                # ambiguous: treat small numbers as millions (e.g., "915" -> 915 triệu)
                return int(val * 1_000_000)
            # else small number -> treat as number (vnd)
            return int(val)
        except:
            return None
    # multiple unitless numbers — unlikely, skip
    # try to find patterns like '1 150' etc; for safety return None
    return None

# Enhanced cleaning: split on last dash, remove engine tokens like 1.6T, 1.6L, AT/MT, drive tokens
ENGINE_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*[tTdDlLvV]?\b") # will catch 1.6T, 1.6L, 2.0 etc
TRANSMISSION_PATTERN = re.compile(r"\b(AT|MT|CVT|DCT|A/T|M/T)\b", flags=re.IGNORECASE)
DRIVE_PATTERN = re.compile(r"\b(4x4|4wd|2wd|4x2|awd|fwd|rwd)\b", flags=re.IGNORECASE)

def clean_name_from_title(raw_title: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """
    Return (clean_name, price_candidate, engine_from_title, drive_from_title)
    uses rsplit('-',1) to isolate trailing price (price_candidate).
    """
    if not raw_title:
        return ("", None, None, None)
    s = normalize_text(raw_title).replace('\u2013','-').replace('\u2014','-')
    # split by last '-' to avoid removing '-' inside model names
    parts = s.rsplit('-', 1)
    if len(parts) == 2:
        left, right = parts
        price_candidate = normalize_text(right)
    else:
        left = parts[0]
        price_candidate = None
    name = normalize_text(left)
    # remove leading 'Xe' etc
    name = re.sub(r'^(xe|bán xe|ban xe)\b[\s:,-]*', '', name, flags=re.IGNORECASE)
    # capture engine token from name before removing it
    engine_match = re.search(r"\b\d+(\.\d+)?[tT]?\b\s*[lL]?\b", name)
    engine_from_title = engine_match.group(0).strip() if engine_match else None
    # capture drive token
    drive_match = DRIVE_PATTERN.search(name)
    drive_from_title = drive_match.group(0).strip() if drive_match else None
    # remove year parentheses / tokens
    name = re.sub(r'\(\s*(19|20)\d{2}\s*\)', '', name)
    name = re.sub(r'\b(19|20)\d{2}\b', '', name)
    # remove engine tokens like 1.6T, 1.6L, 2.0L
    name = re.sub(r'\b\d+(\.\d+)?[tT]?\s*[lL]?\b', '', name)
    # remove transmission tokens
    name = TRANSMISSION_PATTERN.sub('', name)
    # remove drive tokens
    name = DRIVE_PATTERN.sub('', name)
    # remove leftover punctuation and collapse spaces
    name = re.sub(r'[_\-/,:]+', ' ', name)
    name = normalize_text(name)
    return (name, price_candidate, engine_from_title, drive_from_title)

def extract_links_from_list_page(soup: BeautifulSoup) -> List[str]:
    links: List[str] = []
    # try robust selectors
    items = soup.select("li[class*='car-item'], li.car-item")
    for li in items:
        a = li.find('a', itemprop='url') or li.select_one('.cbx a') or li.select_one('a')
        if a and a.get('href'):
            href = a['href']
            if not href.startswith('http'):
                href = 'https://bonbanh.com/' + href.lstrip('/')
            links.append(href)
    if not links:
        for a in soup.select("a[itemprop='url']"):
            href = a.get('href')
            if href:
                if not href.startswith('http'):
                    href = 'https://bonbanh.com/' + href.lstrip('/')
                links.append(href)
    # dedupe while preserving order
    seen = set(); uniq = []
    for u in links:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq

def parse_detail_page(soup: BeautifulSoup, url: str) -> Dict[str,str]:
    out = {k: "" for k in FIELDNAMES}
    out['Link'] = url
    title_tag = soup.select_one('div.title h1') or soup.select_one('h1')
    raw_title = title_tag.get_text(" ", strip=True) if title_tag else ""
    name_clean, price_candidate, engine_from_title, drive_from_title = clean_name_from_title(raw_title)
    out['Ten_xe'] = name_clean

    # price extraction
    price_tag = soup.select_one('.car_price .price') or soup.select_one('span.price')
    price_text = normalize_text(price_tag.get_text(" ", strip=True)) if price_tag else (price_candidate or "")
    out['Gia_ban_text'] = price_text
    num = parse_price_text_to_int(price_text)
    out['Gia_ban'] = str(num) if num is not None else ""

    # detail rows
    rows = soup.select('div.box_car_detail div.row, div.box_car_detail div.row_last')
    if not rows:
        rows = soup.select('div#sgg div.row, div.col div.row')
    for row in rows:
        label_tag = row.select_one('div.label label') or row.find('label')
        value_tag = row.select_one('div.txt_input span.inp') or row.select_one('span.inp') or row.find('span')
        if not label_tag or not value_tag:
            continue
        label = normalize_text(label_tag.get_text(" ", strip=True)).replace(":", "")
        value = normalize_text(value_tag.get_text(" ", strip=True))
        for site_label, field in LABEL_MAP.items():
            if site_label in label:
                out[field] = value
                break

    # fallback for So_cho_ngoi by searching whole page if missing
    if not out['So_cho_ngoi']:
        text = soup.get_text(" ", strip=True)
        m = re.search(r"Số chỗ(?: ngồi)?:\s*([0-9]+(?:\s*chỗ)?)", text, flags=re.IGNORECASE)
        if m:
            out['So_cho_ngoi'] = normalize_text(m.group(1))
        else:
            m2 = re.search(r"\b([1-9][0-9]?)\s*chỗ\b", text, flags=re.IGNORECASE)
            if m2:
                out['So_cho_ngoi'] = m2.group(0)

    # if Dong_co missing, use engine_from_title
    if not out['Dong_co'] and engine_from_title:
        out['Dong_co'] = engine_from_title

    # if Dan_dong missing, use drive_from_title
    if not out['Dan_dong'] and drive_from_title:
        out['Dan_dong'] = drive_from_title

    # fallback year
    if not out['Nam_san_xuat']:
        m = re.search(r"\b(19|20)\d{2}\b", raw_title)
        if m: out['Nam_san_xuat'] = m.group(0)

    # ensure Ten_xe doesn't start with 'Xe '
    out['Ten_xe'] = re.sub(r'^(xe\s+)', '', out['Ten_xe'], flags=re.IGNORECASE).strip()
    return out

def ensure_csv_exists():
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()

def append_row_to_csv(row: Dict[str,str]):
    with open(OUTPUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)

def load_scraped_links() -> Set[str]:
    # now optional (we will not filter by default), keep for compatibility
    links = set()
    if not os.path.exists(OUTPUT_CSV):
        return links
    try:
        with open(OUTPUT_CSV, 'r', encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                link = r.get('Link')
                if link:
                    links.add(link.strip())
    except Exception:
        pass
    return links

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', '-r', action='store_true', help='Delete progress.json and used_cars.csv then start from page1')
    parser.add_argument('--force-start', '-f', action='store_true', help='Start from page1 but keep CSV')
    parser.add_argument('--max-pages', type=int, default=None)
    parser.add_argument('--allow-duplicates', action='store_true', help='Allow duplicates (default true). If specified false, script will filter duplicates')
    args = parser.parse_args()

    # default: allow duplicates (no filtering). If user passes --allow-duplicates=false via wrapper they'd need different flag; here keep simple:
    filter_duplicates = False  # False means do NOT filter duplicates (allow duplicates)
    if not args.allow_duplicates:
        # keep default allow duplicates; user can set flag to explicitly allow - compatibility
        filter_duplicates = False

    if args.reset:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE); print(f"Removed {PROGRESS_FILE}")
        if os.path.exists(OUTPUT_CSV):
            os.remove(OUTPUT_CSV); print(f"Removed {OUTPUT_CSV}")
        start_page = 1
    else:
        progress = load_progress()
        start_page = progress.get('last_page', 1)
        if args.force_start:
            start_page = 1

    print("Starting bonbanh scraper (duplicates allowed by default)...")
    ensure_csv_exists()
    scraped_links = set()
    if filter_duplicates:
        scraped_links = load_scraped_links()

    page = int(start_page or 1)
    consecutive_empty_pages = 0
    max_consecutive_empty = 5
    while True:
        if args.max_pages and page > args.max_pages:
            print("Reached max-pages; stopping.")
            break
        page_url = PAGE_URL_PATTERN.format(page) if page > 1 else START_URL
        print("\n" + "="*50)
        print(f"Processing page {page}: {page_url}")
        resp = safe_get(page_url)
        if not resp:
            consecutive_empty_pages += 1
            print(f"Failed to load page {page}; consecutive_empty={consecutive_empty_pages}")
            if consecutive_empty_pages >= max_consecutive_empty:
                print("Multiple page failures; stopping.")
                break
            page += 1
            time.sleep(random.uniform(*PAGE_DELAY))
            continue
        soup = BeautifulSoup(resp.text, 'html.parser')
        links = extract_links_from_list_page(soup)
        if filter_duplicates:
            links = [l for l in links if l not in scraped_links]
        if not links:
            print(f"No new car items found on page {page}.")
            consecutive_empty_pages += 1
            if consecutive_empty_pages >= max_consecutive_empty:
                print("No items for several pages; stopping.")
                break
            page += 1
            time.sleep(random.uniform(*PAGE_DELAY))
            continue
        consecutive_empty_pages = 0
        print(f"Found {len(links)} items on page {page} (dup filtering={'ON' if filter_duplicates else 'OFF'}).")
        for link in tqdm(links, desc=f"Page {page}", unit='car'):
            detail = safe_get(link)
            if not detail:
                print(f"Skipping {link} due to load failure.")
                continue
            try:
                row = parse_detail_page(BeautifulSoup(detail.text, 'html.parser'), link)
            except Exception as e:
                print(f"Parse error {link}: {e}")
                continue
            append_row_to_csv(row)
            if filter_duplicates:
                scraped_links.add(link)
            save_progress({'last_page': page})
            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        print(f"Finished page {page}. Wrote {len(links)} items.")
        save_progress({'last_page': page})
        page += 1
        time.sleep(random.uniform(*PAGE_DELAY))
    print("Done.")

if __name__ == '__main__':
    main()
