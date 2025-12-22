# data_standardization.py
# DATA STANDARDIZATION PIPELINE (ETL STEP 1)
# QUY TRÌNH CHUẨN HÓA DỮ LIỆU
#
# Description:
# 1. Read raw JSON data / Đọc dữ liệu thô từ JSON.
# 2. Normalize Brand names / Chuẩn hóa tên thương hiệu.
# 3. Extract technical specs using Regex / Trích xuất thông số kỹ thuật.
# 4. Handle data anomalies safely / Xử lý an toàn các dữ liệu lỗi.
#
# Author: AI Engineer
# Date: 2026

import json
import re
from pathlib import Path
from typing import Dict, Optional, Any

# 1. PATH CONFIGURATION (CẤU HÌNH ĐƯỜNG DẪN)
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

INPUT_FILE = DATA_DIR / "datalaptop.json"
OUTPUT_FILE = DATA_DIR / "products_final.jsonl"

# Ensure output directory exists / Đảm bảo thư mục tồn tại
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# 2. UTILITY FUNCTIONS (CÁC HÀM TIỆN ÍCH)
def norm(text: Optional[str]) -> str:
    """
    Normalize text: lower case and strip whitespace.
    Chuẩn hóa văn bản: chuyển thường và xóa khoảng trắng thừa.
    """
    return text.lower().strip() if isinstance(text, str) else ""

def extract_int(pattern: str, text: str) -> Optional[int]:
    """
    Extract integer safely using Regex.
    Trích xuất số nguyên an toàn.
    """
    if not text: return None
    # Use IGNORECASE for robustness / Dùng cờ IGNORECASE để bắt cả hoa/thường
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def extract_float(pattern: str, text: str) -> Optional[float]:
    """
    Extract float safely using Regex.
    Trích xuất số thực an toàn (xử lý dấu phẩy).
    """
    if not text: return None
    text = text.replace(",", ".")
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None

# 3. BRAND NORMALIZATION (CHUẨN HÓA THƯƠNG HIỆU)
def normalize_brand(name: str) -> str:
    """
    Map various brand spellings to canonical names.
    Ánh xạ các cách viết tên hãng về tên chuẩn.
    """
    t = norm(name)
    
    if "gigabyte" in t or "giga" in t: return "Gigabyte"
    if "asus" in t: return "Asus"
    if "msi" in t: return "MSI"
    if "acer" in t: return "Acer"
    if "lenovo" in t: return "Lenovo"
    if "dell" in t: return "Dell"
    if "hp" in t: return "HP"
    if "macbook" in t or "apple" in t: return "Apple"
    if "lg" in t: return "LG"
    
    return "Other"

# 4. CORE STANDARDIZATION LOGIC (LOGIC CHUẨN HÓA CHÍNH)
def standardize(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """
    Process a single raw product item into a structured format.
    Xử lý một sản phẩm thô thành định dạng cấu trúc chuẩn.
    """
    # Fallback name if missing / Tên mặc định nếu thiếu
    name = item.get("Tên sản phẩm", f"Laptop {idx}")

    # 1. Price Parsing / Xử lý giá tiền
    # Handle 'Contact' or invalid prices gracefully
    try:
        price_str = str(item.get("Giá", 0))
        price = int(re.sub(r"[^\d]", "", price_str))
    except ValueError:
        price = 0

    # 2. Raw Specs / Thông số thô
    cpu = item.get("Công nghệ CPU", "")
    gpu = item.get("Card màn hình", "")

    # 3. RAM Parsing / Xử lý RAM
    # Captures: 16GB, 16 gb, 16Gb...
    ram_raw = norm(item.get("RAM", ""))
    ram = extract_int(r"(\d+)\s*gb", ram_raw) or 0

    # 4. Storage Parsing / Xử lý ổ cứng
    # Fix regex group issue: use non-capturing group (?:...)
    # Bắt số đứng trước GB hoặc TB
    ssd_raw = norm(item.get("Ổ cứng", ""))
    ssd = extract_int(r"(\d+)\s*(?:gb|tb)", ssd_raw) or 0
    
    # Convert TB to GB / Đổi TB sang GB
    if "tb" in ssd_raw:
        ssd *= 1024

    # 5. Screen Size / Kích thước màn hình
    screen = extract_float(r"(\d+(\.\d+)?)", item.get("Kích thước màn hình", "")) or 0.0

    # 6. Refresh Rate / Tần số quét
    hz_raw = norm(item.get("Tần số quét", ""))
    hz = extract_int(r"(\d+)\s*hz", hz_raw)
    
    # Heuristic fallback: if no 'Hz' unit found, check for common values > 50
    if not hz:
        hz_fallback = extract_int(r"\b(\d{2,3})\b", hz_raw)
        hz = hz_fallback if hz_fallback and hz_fallback > 50 else 60

    # 7. Weight / Trọng lượng
    # Default to 0.0 to allow math operations later
    weight = extract_float(r"(\d+(\.\d+)?)\s*kg", norm(item.get("Kích thước", ""))) or 0.0

    # 8. Brand / Thương hiệu
    brand = normalize_brand(name)

    # Return structured dict / Trả về dictionary đã chuẩn hóa
    return {
        "id": idx,
        "name": name,
        "brand": brand,
        "price_value": price,
        "cpu": cpu,
        "gpu": gpu,
        "ram_gb": ram,
        "ssd_gb": ssd,
        "screen_size_inch": screen,
        "refresh_rate_hz": hz,
        "weight_kg": weight,
        "raw_source": item,  # Keep full raw data / Giữ lại toàn bộ dữ liệu gốc
    }

# 5. MAIN EXECUTION (CHƯƠNG TRÌNH CHÍNH)
def main():
    if not INPUT_FILE.exists():
        print(f"❌ Input file not found / Không tìm thấy file: {INPUT_FILE}")
        return

    print(f"📂 Reading data from / Đang đọc dữ liệu từ: {INPUT_FILE}")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"🔄 Standardizing {len(data)} products... / Đang chuẩn hóa...")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, item in enumerate(data, 1):
            std_item = standardize(item, i)
            # Write line by line (JSONL) / Ghi từng dòng format JSONL
            f.write(json.dumps(std_item, ensure_ascii=False) + "\n")

    print(f"✅ Standardization complete! / Hoàn tất chuẩn hóa!")
    print(f"📄 Output saved to / File lưu tại: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()