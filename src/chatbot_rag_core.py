import os
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from chromadb import PersistentClient
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# 1. CONFIGURATION & CONSTANTS
# Configuration and constants for the RAG system.
# Cấu hình và hằng số cho hệ thống RAG.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_DIR = DATA_DIR / "chroma_db"
LOG_FILE = DATA_DIR / "user_logs.jsonl"

COLLECTION_NAME = "laptops"
EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768

DEFAULT_TOP_K = 250  # Number of initial candidates to retrieve / Số ứng viên ban đầu để lấy.
DEFAULT_RETURN_K = 5  # Number of top results to return / Số kết quả hàng đầu để trả về.
SCAN_BATCH = 5000  # Batch size for scanning the database / Kích thước lô để quét cơ sở dữ liệu.

# 2. ENVIRONMENT INITIALIZATION
# Load environment variables and configure APIs.
# Tải biến môi trường và cấu hình API.
load_dotenv(ROOT / ".env")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# 3. TEXT UTILITIES
# Utility functions for text normalization.
# Các hàm tiện ích để chuẩn hóa văn bản.
def remove_accents(input_str: str) -> str:
    """
    Normalize Vietnamese text by removing accents and handling 'đ' -> 'd'.
    Chuẩn hóa văn bản tiếng Việt bằng cách loại bỏ dấu và xử lý 'đ' -> 'd'.
    """
    if not input_str:
        return ""
    input_str = input_str.replace("đ", "d").replace("Đ", "D")
    s1 = unicodedata.normalize('NFD', input_str)
    s2 = ''.join(c for c in s1 if unicodedata.category(c) != 'Mn')
    return s2.lower()

def norm_text(s: str) -> str:
    """
    Standardize input text: lowercase, remove accents, clean spaces.
    Chuẩn hóa văn bản đầu vào: chữ thường, loại bỏ dấu, làm sạch khoảng trắng.
    """
    s = (s or "").strip().lower()
    s = remove_accents(s)
    s = re.sub(r"\s+", " ", s)
    return s

# 4. INTENT DETECTION
# Functions for detecting user intent from query.
# Các hàm để phát hiện ý định người dùng từ truy vấn.
def detect_purpose_from_query(query: str) -> Optional[str]:
    """
    Detect user intent based on keywords in Vietnamese queries.
    Phát hiện ý định người dùng dựa trên từ khóa trong truy vấn tiếng Việt.
    """
    q_norm = norm_text(query)
    
    # Gaming keywords / Từ khóa cho gaming
    gaming_keywords = [
        'game', 'choi', 'lol', 'pubg', 'valorant', 'csgo', 'genshin', 'gta', 'steam', 'gaming', 'fifa',
        'choi game', 'chien game', 'rtx', 'gtx', 'card roi', 'card do hoa roi', 'fps', 'do hoa manh'
    ]
    if any(k in q_norm for k in gaming_keywords):
        return "gaming"
    
    # Creator keywords (design, editing, programming) / Từ khóa cho creator (thiết kế, chỉnh sửa, lập trình)
    creator_keywords = [
        'do hoa', 'thiet ke', 'render', 'dung phim', 'edit', 'video', 'adobe', 'photoshop', 'premiere',
        '3d', 'cad', 'revit', 'ai', 'kien truc', 'lap trinh', 'code', 'programming', 'blender', 'after effects',
        'autocad', 'solidworks', 'illustrator', 'maya'
    ]
    if any(k in q_norm for k in creator_keywords):
        return "creator"
    
    # Thin & Light keywords (portability, battery life) / Từ khóa cho thinlight (mỏng nhẹ, pin trâu)
    thin_keywords = [
        'mong', 'nhe', 'di dong', 'doanh nhan', 'macbook', 'air', 'pin trau', 'pin lau', 'gon', 'sang chanh',
        'ultrabook', 'mang di', 'di chuyen', 'nho gon', 'nhe nhang', 'pin dai'
    ]
    if any(k in q_norm for k in thin_keywords):
        return "thinlight"
    
    # Office keywords (work, study) / Từ khóa cho office (văn phòng, học tập)
    office_keywords = [
        'van phong', 'hoc tap', 'sinh vien', 'word', 'excel', 'powerpoint', 'ke toan', 'giao vien',
        'office', 'lam viec', 'hoc online', 'zoom', 'teams'
    ]
    if any(k in q_norm for k in office_keywords):
        return "office"

    # Premium screen keywords (prioritize creator) / Từ khóa màn hình cao cấp (ưu tiên creator)
    premium_screen_keywords = ['oled', 'man oled', 'man dep', 'man sac net', '120hz', '144hz', '165hz', 'mini led', 'retina']
    if any(k in q_norm for k in premium_screen_keywords):
        return "creator"

    return None

# 5. PRICE EXTRACTION
# Function for extracting price range from query.
# Hàm trích xuất khoảng giá từ truy vấn.
def extract_price_range(query: str) -> Tuple[int, int, bool]:
    """
    Extract price range from user query in Vietnamese.
    
    Supports common formats:
    - Range: "15-25tr", "từ 20 đến 30 triệu", "15 đến 25tr"
    - Single: "30 triệu", "khoảng 25tr", "dưới 40 triệu", "trên 20tr"
    - Units: tr, triệu, m (defaults to triệu if no unit)
    
    Args:
        query (str): User's query or requirement.
    
    Returns:
        Tuple[int, int, bool]: 
            - min_price: Lowest price (VND)
            - max_price: Highest price (VND)
            - has_price: True if price info detected, False otherwise
    """
    if not query:
        return 0, 100_000_000, False

    t = norm_text(query).replace(",", ".")  # Normalize: 30,5 → 30.5
    # Chuẩn hóa: 30,5 → 30.5

    # 1. Clear range: 15-25tr, 15 to 25 million, from 20 to 30tr
    # Khoảng giá rõ ràng: 15-25tr, 15 đến 25 triệu, từ 20 tới 30tr
    range_pat = r"(\d+(?:\.\d+)?)\s*(?:-|–|đến|den|toi|tới)\s*(\d+(?:\.\d+)?)\s*(tr|triệu|m)?"
    m = re.search(range_pat, t)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        unit = m.group(3) or "triệu"  # Default to triệu if no unit / Mặc định triệu nếu không có đơn vị
        mul = 1_000_000 if unit in ["tr", "triệu", "m"] else 1
        return int(low * mul), int(high * mul), True

    # 2. Range style "from ... to ..."
    # Khoảng giá kiểu "từ ... đến ..."
    from_to_pat = r"(?:từ|khoảng)\s*(\d+(?:\.\d+)?)\s*(?:đến|tới|den)\s*(\d+(?:\.\d+)?)\s*(tr|triệu|m)?"
    m = re.search(from_to_pat, t)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        unit = m.group(3) or "triệu"
        mul = 1_000_000 if unit in ["tr", "triệu", "m"] else 1
        return int(low * mul), int(high * mul), True

    # 3. Single price: 30 million, about 25tr, under 40 million, over 15tr, 30tr
    # Giá đơn: 30 triệu, khoảng 25tr, dưới 40 triệu, trên 15tr, 25tr
    single_pat = r"(?:khoảng|khoang|cir|dưới|duoi|trên|tren)?\s*(\d+(?:\.\d+)?)\s*(tr|triệu|m|k)?"
    m = re.search(single_pat, t)
    if m:
        num_str = m.group(1)
        unit = m.group(2) or "triệu"  # Default to triệu if no unit / Mặc định triệu nếu không có đơn vị

        try:
            x = float(num_str)
        except ValueError:
            return 0, 100_000_000, False

        mul = 1_000_000 if unit in ["tr", "triệu", "m"] else (1_000 if unit == "k" else 1)
        x *= mul

        full_match = m.group(0).lower()
        if "dưới" in full_match or "duoi" in full_match:
            return 0, int(x * 1.05), True
        elif "trên" in full_match or "tren" in full_match:
            return int(x * 0.95), 100_000_000, True
        else:
            expand = 500_000 if x < 10_000_000 else 1_000_000  # Reasonable expansion based on budget / Mở rộng hợp lý dựa trên ngân sách
            return int(max(0, x - expand)), int(x + expand), True

    return 0, 100_000_000, False

# 6. BRAND HANDLING
# Functions for handling brand names and matching.
# Các hàm xử lý tên thương hiệu và khớp.
BRAND_SYNONYMS = {
    "asus": ["asus"], "acer": ["acer"], "dell": ["dell"], "hp": ["hp", "hewlett packard"],
    "lenovo": ["lenovo"], "msi": ["msi"], "apple": ["apple", "macbook", "mac"], "gigabyte": ["gigabyte", "giga"],
}

def canonicalize_brands(brands_ui: Optional[List[str]]) -> List[str]:
    """
    Canonicalize brand names from user input.
    Chuẩn hóa tên thương hiệu từ đầu vào người dùng.
    """
    if not brands_ui: return []
    out = []
    for b in brands_ui:
        key = norm_text(b)
        for canon, syns in BRAND_SYNONYMS.items():
            if key == canon or key in syns:
                out.append(canon)
                break
    return list(set(out))

def detect_brand_from_query(query: str) -> Optional[str]:
    """
    Detect brand from query.
    Phát hiện thương hiệu từ truy vấn.
    """
    t = norm_text(query)
    for brand, keys in BRAND_SYNONYMS.items():
        if any(k in t for k in keys): return brand
    return None

def item_matches_brand(item: Dict[str, Any], wanted_canons: List[str]) -> bool:
    """
    Check if item matches wanted brands.
    Kiểm tra xem sản phẩm có khớp với các thương hiệu mong muốn.
    """
    if not wanted_canons: return True
    b_norm = norm_text(str(item.get("brand", "")))
    name_norm = norm_text(str(item.get("name", "")))
    
    raw = item.get("raw_source", {})
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except: raw = {}
    raw_name_norm = norm_text(str(raw.get("Tên sản phẩm", "")))

    hay = f"{b_norm} {name_norm} {raw_name_norm}"
    for canon in wanted_canons:
        keys = BRAND_SYNONYMS.get(canon, [canon])
        if any(k in hay for k in keys): return True
    return False

# 7. VECTOR SEARCH & EMBEDDING
# Functions for vector search and embedding.
# Các hàm cho tìm kiếm vector và embedding.
def safe_embed(text: str, task_type: str) -> List[float]:
    """
    Safely generate embedding with retry logic.
    Tạo embedding an toàn với cơ chế thử lại.
    """
    for _ in range(3):
        try:
            resp = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
            emb = resp["embedding"]
            if len(emb) < EMBED_DIM: emb += [0.0] * (EMBED_DIM - len(emb))
            elif len(emb) > EMBED_DIM: emb = emb[:EMBED_DIM]
            return emb
        except Exception: pass
    return [0.0] * EMBED_DIM

_client = PersistentClient(path=str(DB_DIR))
try: _collection = _client.get_collection(COLLECTION_NAME)
except Exception: _collection = None

def vector_search(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    """
    Perform vector search on the query.
    Thực hiện tìm kiếm vector trên truy vấn.
    """
    if _collection is None: return []
    emb = safe_embed(query, task_type="retrieval_query")
    try:
        res = _collection.query(query_embeddings=[emb], n_results=top_k)
        return res.get("metadatas", [[]])[0] or []
    except Exception: return []

# 8. DATA PARSING & ENRICHMENT
# Functions for parsing and enriching item data.
# Các hàm phân tích và làm giàu dữ liệu sản phẩm.
def _to_int(x, default=0) -> int:
    """
    Safely convert to integer.
    Chuyển đổi an toàn sang số nguyên.
    """
    try: return int(float(str(x).strip()))
    except: return default

def _to_float(x, default=0.0) -> float:
    """
    Safely convert to float.
    Chuyển đổi an toàn sang số thực.
    """
    try: return float(str(x).strip())
    except: return default

def _parse_gb_from_text(s: str) -> int:
    """
    Parse GB/TB from text.
    Phân tích GB/TB từ văn bản.
    """
    if not s: return 0
    t = norm_text(s).replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*tb\b", t)
    if m: return int(float(m.group(1)) * 1024)
    m = re.search(r"(\d+(?:\.\d+)?)\s*gb\b", t)
    if m: return int(float(m.group(1)))
    return 0

def _parse_screen_inch(s: str) -> float:
    """
    Parse screen size in inches.
    Phân tích kích thước màn hình theo inch.
    """
    if not s: return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)", str(s))
    return float(m.group(1)) if m else 0.0

def _parse_hz(s: str) -> int:
    """
    Parse refresh rate in Hz.
    Phân tích tần số quét theo Hz.
    """
    if not s: return 0
    m = re.search(r"(\d+)\s*hz", str(s).lower())
    return int(m.group(1)) if m else 0

def _parse_weight_kg(raw: Dict[str, Any]) -> float:
    """
    Parse weight in kg from raw data.
    Phân tích trọng lượng theo kg từ dữ liệu thô.
    """
    for k in ["Kích thước", "Khối lượng", "Trong luong"]:
        s = str(raw.get(k, "") or "").lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*kg", s)
        if m: return float(m.group(1))
    return 0.0

def enrich_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and enrich product metadata.
    Chuẩn hóa và làm giàu metadata sản phẩm.
    """
    out = dict(item)
    raw = out.get("raw_source", {})
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except: raw = {}
    out["raw_source"] = raw

    if not out.get("name"): out["name"] = raw.get("Tên sản phẩm", "")
    if not out.get("brand"):
        n = norm_text(out.get("name", ""))
        for canon, keys in BRAND_SYNONYMS.items():
            if any(k in n for k in keys):
                out["brand"] = canon
                break
    else: out["brand"] = norm_text(str(out["brand"]))

    out["price_value"] = _to_int(out.get("price_value") or raw.get("Giá", 0))

    if not out.get("cpu"): out["cpu"] = raw.get("Công nghệ CPU", "")
    if not out.get("gpu"): out["gpu"] = raw.get("Card màn hình", "")

    if _to_int(out.get("ram_gb")) <= 0: out["ram_gb"] = _parse_gb_from_text(str(raw.get("RAM", "")))
    if _to_int(out.get("ssd_gb")) <= 0: out["ssd_gb"] = _parse_gb_from_text(str(raw.get("Ổ cứng", "")))
    if _to_float(out.get("screen_size_inch")) <= 0: out["screen_size_inch"] = _parse_screen_inch(str(raw.get("Kích thước màn hình", "")))
    if _to_int(out.get("refresh_rate_hz")) <= 0: out["refresh_rate_hz"] = _parse_hz(str(raw.get("Tần số quét", "")))
    if _to_float(out.get("weight_kg")) <= 0: out["weight_kg"] = _parse_weight_kg(raw)

    raw_screen = norm_text(str(raw.get("Công nghệ màn hình", "") + raw.get("Màn hình", "") + raw.get("Độ phân giải", "")))
    out["is_ips"] = "ips" in raw_screen
    out["is_oled"] = "oled" in raw_screen
    out["is_high_res"] = any(x in raw_screen for x in ["2k", "4k", "qhd", "retina", "2.8k", "3k"])
    
    return out

# 8. HARDWARE LOGIC
# Logic for hardware classification and expectations.
# Logic cho phân loại phần cứng và kỳ vọng.
def get_price_segment(price):
    """
    Classify product into price segments.
    Phân loại sản phẩm vào các phân khúc giá.
    """
    if price < 15_000_000: return "Budget"
    if price < 25_000_000: return "Mid"
    if price < 35_000_000: return "Upper-Mid"
    if price < 50_000_000: return "High"
    if price < 80_000_000: return "Premium"
    return "Ultra"

def get_gpu_expectation(segment):
    """
    Get minimum expected GPU tier for a price segment.
    Lấy cấp độ GPU kỳ vọng tối thiểu cho phân khúc giá.
    """
    if segment == 'Budget': return 3.0
    if segment == 'Mid': return 6.0 
    if segment == 'Upper-Mid': return 7.5
    if segment == 'High': return 8.0 
    if segment == 'Premium': return 8.5 
    if segment == 'Ultra': return 9.5
    return 3.0

def get_gpu_tier(gpu_str: str, is_apple: bool, intent: str) -> float:
    """
    Advanced GPU tier classification (0-10), updated with RTX 5000 series.
    Phân loại cấp độ GPU nâng cao (0-10), cập nhật series RTX 5000.
    """
    g = str(gpu_str).upper()
    
    # Apple GPU logic / Logic GPU Apple
    if is_apple:
        if intent == 'creator':
            if 'MAX' in g: return 8.5
            if 'PRO' in g: return 7.5
            return 6.5
        if 'MAX' in g: return 7.0
        if 'PRO' in g: return 6.0
        return 4.5 

    # NVIDIA discrete (RTX 5000 series) / NVIDIA rời (series RTX 5000)
    if '5090' in g: return 10.0
    if '5080' in g: return 9.6
    if '5070' in g: return 8.8
    if '5060' in g: return 8.2
    if '5050' in g: return 7.5

    # NVIDIA discrete (4000/3000 series) / NVIDIA rời (series 4000/3000)
    if '4090' in g: return 9.8
    if '4080' in g: return 9.3
    if '4070' in g: return 8.5
    if '4060' in g: return 7.5
    if '4050' in g: return 7.0
    if '3050' in g: return 6.0
    if '2050' in g or 'GTX' in g or '1650' in g: return 5.0
    
    # AMD & integrated GPUs / AMD và GPU tích hợp
    if 'RX 7600' in g or 'RX 6700' in g: return 7.0
    if 'RX 6600' in g: return 6.5
    if 'RX 6500' in g: return 5.5
    
    if 'ARC' in g: return 5.5 if intent != 'gaming' else 4.5
    if '780M' in g or '680M' in g or '880M' in g or '890M' in g: return 6.0
    if 'ULTRA' in g: return 5.0
    if 'VEGA' in g or 'RADEON' in g: return 4.0
    if 'IRIS' in g or 'XE' in g or 'UHD' in g: return 3.0
    
    return 2.0

# 9. FILTERING & SCORING
# Functions for filtering and scoring products.
# Các hàm lọc và chấm điểm sản phẩm.
def filter_by_price(items: List[Dict[str, Any]], low: int, high: int) -> List[Dict[str, Any]]:
    """
    Filter items by price range.
    Lọc sản phẩm theo khoảng giá.
    """
    out = []
    for it in items:
        price = _to_int(it.get("price_value", 0), 0)
        if low <= price <= high:
            out.append(it)
    return out

def filter_by_purpose_strict(items: List[Dict[str, Any]], purpose: str) -> List[Dict[str, Any]]:
    """
    Strictly filter products based on user purpose.
    
    Detailed logic:
    - Gaming: Prioritize products with discrete GPUs or gaming-related names.
    - Office: Exclude products with discrete GPUs or gaming names to focus on lightweight, integrated graphics models.
    - Creator: Require at least 16GB RAM and prefer discrete GPUs for heavy tasks.
    - Thinlight: Limit to lightweight products (under 1.8kg) for portability.
    
    This ensures relevant results by excluding mismatched categories, e.g., no gaming laptops for office use.
    
    Lọc sản phẩm nghiêm ngặt dựa trên mục đích người dùng.
    
    Logic chi tiết:
    - Gaming: Ưu tiên sản phẩm có GPU rời hoặc tên liên quan đến gaming.
    - Office: Loại bỏ sản phẩm có GPU rời hoặc tên gaming để tập trung vào máy nhẹ, GPU tích hợp.
    - Creator: Yêu cầu ít nhất 16GB RAM và ưu tiên GPU rời cho công việc nặng.
    - Thinlight: Giới hạn sản phẩm nhẹ (dưới 1.8kg) để ưu tiên tính di động.
    
    Đảm bảo kết quả phù hợp bằng cách loại bỏ các hạng mục không khớp, ví dụ không gợi ý laptop gaming cho văn phòng.
    """
    p = (purpose or "").lower().strip()
    if not p:
        return items

    out = []
    for it in items:
        gpu = norm_text(it.get("gpu", ""))
        name = norm_text(it.get("name", ""))
        weight = _to_float(it.get("weight_kg", 0.0), 0.0)
        
        has_dgpu = any(x in gpu for x in ["rtx", "gtx", "rx ", "geforce", "radeon rx", "nvidia rtx"])
        is_gaming_name = any(x in name for x in ["gaming", "loq", "legion", "rog", "nitro", "predator", "tuf"])

        if p == "gaming":
            if has_dgpu or is_gaming_name or "game" in name:
                out.append(it)
        
        elif p == "office":
            # Exclude gaming-related products for pure office focus
            # Loại bỏ sản phẩm liên quan gaming để tập trung văn phòng thuần túy
            if not has_dgpu and not is_gaming_name:
                out.append(it)
        
        elif p == "thinlight":
            if (0 < weight <= 1.8) or weight == 0:  # Allow unknown weight but prefer lightweight
                # Cho phép trọng lượng không biết nhưng ưu tiên máy nhẹ
                out.append(it)
        
        elif p == "creator":
            if it.get("ram_gb", 0) >= 16 or has_dgpu:
                out.append(it)
        
        else:
            out.append(it)
    return out

def calculate_score(item: Dict, purpose: str, rank_index: int = 0) -> float:
    """
    Calculate product match score with rank decay.
    
    Unified scoring logic including purpose-specific weights and adjustments.
    Chấm điểm phù hợp sản phẩm với giảm điểm theo thứ hạng.
    
    Logic chấm điểm thống nhất bao gồm trọng số theo mục đích và điều chỉnh.
    """
    raw = item.get("raw_source", {})
    if isinstance(raw, str): raw = json.loads(raw) if raw else {}
    
    brand = str(item.get('brand', '')).upper()
    price = float(item.get('price_value', 0))
    gpu_str = str(item.get('gpu', '')).upper()
    cpu_str = str(item.get('cpu', '')).upper()
    screen_str = str(raw.get('Màn hình', '')).upper()
    ram = int(item.get('ram_gb', 8))
    ssd = int(item.get('ssd_gb', 256))
    refresh_rate = int(item.get('refresh_rate_hz', 60))
    weight = float(item.get('weight_kg', 2.0))

    segment = get_price_segment(price)
    is_apple = 'APPLE' in brand or any(x in cpu_str for x in ['M1', 'M2', 'M3', 'M4'])
    is_intel_ultra = 'ULTRA' in cpu_str
    
    gpu_tier = get_gpu_tier(gpu_str, is_apple, purpose)
    
    is_weak_igpu = gpu_tier <= 4.0 and not ('RTX' in gpu_str or 'GTX' in gpu_str or 'RX' in gpu_str)
    is_igpu_general = gpu_tier < 6.0 and not ('RTX' in gpu_str or 'GTX' in gpu_str)
    if is_apple:
        is_weak_igpu = False
        is_igpu_general = True

    # Gaming Desk Logic
    # Logic máy gaming bàn (nặng nhưng mạnh)
    is_gaming_desk = purpose == 'gaming' and weight >= 2.3

    score_gpu = gpu_tier * 10
    if purpose == 'gaming' and is_weak_igpu and not is_apple: score_gpu = 25

    score_cpu = 60
    if is_apple: score_cpu = 90
    elif is_intel_ultra: score_cpu = 85
    elif 'I9' in cpu_str or 'R9' in cpu_str: score_cpu = 90
    elif 'I7' in cpu_str or 'R7' in cpu_str: score_cpu = 80
    elif 'I5' in cpu_str or 'R5' in cpu_str: score_cpu = 70
    elif 'I3' in cpu_str: score_cpu = 50

    score_mem = 80
    if ram < 8: score_mem = 30
    elif ram >= 16: score_mem = 100
    if price > 20_000_000 and ssd < 512: score_mem -= 20

    score_form = 70
    if 'OLED' in screen_str or 'RETINA' in screen_str or 'DCI-P3' in screen_str: score_form += 15
    if '2K' in screen_str or '4K' in screen_str or 'QHD' in screen_str: score_form += 10
    
    if purpose == 'gaming':
        if refresh_rate >= 144: score_form += 20
        elif refresh_rate < 120 and not is_apple: score_form -= 15
        
    if purpose == 'thinlight':
        if weight < 1.3: score_form = 100
        elif weight < 1.5: score_form = 90
        elif weight > 1.8: score_form = 40
        elif weight > 2.2: score_form = 20

    score_value = 80
    expected_gpu = get_gpu_expectation(segment)
    
    if ram < 16 and segment in ['High', 'Premium', 'Ultra']: score_value -= 15
    if ssd < 512 and segment != 'Budget': score_value -= 10
    
    if is_apple:
        if segment == 'Ultra' and 'MAX' not in cpu_str: score_value -= 20
        elif segment == 'Premium' and 'PRO' not in cpu_str: score_value -= 10
    else:
        diff = gpu_tier - expected_gpu
        if diff >= 1.0: score_value += 15 
        elif diff <= -2.0: score_value -= 40
        elif diff <= -1.0: score_value -= 25

    final_score = 0
    if purpose == 'gaming':
        final_score = (score_gpu * 0.32) + (score_value * 0.28) + (score_cpu * 0.15) + (score_mem * 0.1) + (score_form * 0.15)
        if is_weak_igpu and not is_apple: final_score -= 35
        elif is_igpu_general and not is_apple: final_score -= 15
        if is_apple: final_score -= 35
        if is_gaming_desk: final_score += 6 

    elif purpose == 'creator':
        final_score = (score_value * 0.3) + (score_gpu * 0.25) + (score_cpu * 0.2) + (score_mem * 0.1) + (score_form * 0.15)
        if is_apple: final_score += 5
        if is_weak_igpu and not is_apple: final_score -= 15
        
    elif purpose == 'office':
        final_score = (score_value * 0.4) + (score_cpu * 0.2) + (score_mem * 0.2) + (score_form * 0.15) + (score_gpu * 0.05)
        
    elif purpose == 'thinlight':
        final_score = (score_form * 0.4) + (score_value * 0.3) + (score_cpu * 0.15) + (score_mem * 0.1) + (score_gpu * 0.05)
        if weight > 2.0: final_score -= 15
        if is_apple: final_score += 5
    
    # Rank decay
    # Giảm điểm theo thứ hạng
    if rank_index == 0: decay = 0
    elif rank_index == 1: decay = 0.8
    elif rank_index == 2: decay = 1.6
    else: decay = rank_index * 0.7
    
    final_score = final_score - decay
    min_score = 35.0 if purpose == 'gaming' else 50.5
    return min(99.5, max(min_score, final_score))

def rerank(items: List[Dict[str, Any]], purpose: str) -> List[Dict[str, Any]]:
    """
    Rerank items based on score.
    Xếp hạng lại sản phẩm dựa trên điểm số.
    """
    for it in items:
        it["fit_score"] = round(calculate_score(it, purpose), 1)
    
    def key(it: Dict[str, Any]):
        return -it.get("fit_score", 0.0)
    
    return sorted(items, key=key)

# 10. LLM GENERATION
# Generate advice using LLM.
# Tạo lời khuyên sử dụng LLM.
def generate_llm_advice(query: str, purpose: str, products: List[Dict[str, Any]]) -> str:
    """
    Generate professional advice using Groq LLM.
    Tạo lời khuyên chuyên nghiệp sử dụng Groq LLM.
    """
    if not products:
        return "Mình chưa tìm được mẫu laptop phù hợp trong tầm giá này. Bạn thử tăng ngân sách hoặc đổi nhu cầu nhé."

    lines = []
    for i, p in enumerate(products[:3], 1):
        score = p.get('fit_score', 0)
        line = (
            f"{i}. {p.get('name')} (Độ phù hợp: {score}%)\n"
            f"   - Giá: {p.get('price_value'):,} VNĐ\n"
            f"   - CPU: {p.get('cpu')} | GPU: {p.get('gpu')}\n"
            f"   - RAM: {p.get('ram_gb')}GB | SSD: {p.get('ssd_gb')}GB | Nặng: {p.get('weight_kg')}kg\n"
        )
        lines.append(line)
    
    product_text = "\n".join(lines)
    
    prompt = f"""
    Bạn là chuyên gia tư vấn laptop thân thiện, nói chuyện như đang tư vấn trực tiếp cho khách.

    Khách hỏi: "{query}" (mục đích chính: {purpose})

    Top 3 máy phù hợp nhất (xếp theo độ phù hợp giảm dần):
    {product_text}

    Hãy trả lời ngắn gọn 4-6 câu:
    - Khuyên dùng máy số 1 và giải thích tại sao nó phù hợp nhất.
    - So sánh nhanh với máy số 2 (nếu có điểm khác biệt rõ rệt).
    - Kết luận rõ nên chọn máy nào trong hầu hết trường hợp.

    Nói tự nhiên, không liệt kê thông số khô khan, không nhắc đến điểm số hay hệ thống.
    """

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý ảo tư vấn laptop chuyên nghiệp, ngắn gọn."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=450,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Groq API Error: {e}")
        return "Hệ thống đã tìm thấy các sản phẩm phù hợp nhất. Mời bạn xem chi tiết bên dưới."

# 11. SCAN & LOG
# Functions for scanning database and logging.
# Các hàm quét cơ sở dữ liệu và ghi log.
def scan_db_by_price_brand(low: int, high: int, wanted_canons: List[str], need: int) -> List[Dict[str, Any]]:
    """
    Scan database for products in price range and brands.
    Quét cơ sở dữ liệu cho sản phẩm trong khoảng giá và thương hiệu.
    """
    if _collection is None: return []
    found: List[Dict[str, Any]] = []
    offset = 0
    while True:
        try:
            res = _collection.get(include=["metadatas"], limit=SCAN_BATCH, offset=offset)
            metas = res.get("metadatas", []) or []
        except Exception: break
        if not metas: break

        for m in metas:
            it = enrich_item(m)
            price = _to_int(it.get("price_value", 0), 0)
            if price < low or price > high: continue
            if wanted_canons and not item_matches_brand(it, wanted_canons): continue
            found.append(it)
        
        if len(found) >= max(need, DEFAULT_RETURN_K): break
        offset += len(metas)
    return found

def log_user_interaction(query: str, purpose: str, price_range: Tuple[int, int], advice: str, products: List[Dict]):
    """
    Log user interaction to file.
    Ghi log tương tác người dùng vào file.
    """
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "intent": purpose,
            "price_min": price_range[0],
            "price_max": price_range[1],
            "ai_response": advice[:200] + "..." if len(advice) > 200 else advice,
            "recommended_models": [p.get("name", "Unknown") for p in products[:5]]
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Logger Error: {e}")

# 12. MAIN PUBLIC INTERFACE
# Main function to get answer from query.
# Hàm chính để lấy câu trả lời từ truy vấn.
def get_answer(
    query: str,
    purpose: str,
    expand_million: int,
    brands: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    return_k: int = DEFAULT_RETURN_K,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Orchestrate the RAG pipeline to generate answer and products.
    Điều phối toàn bộ quy trình RAG để tạo câu trả lời và sản phẩm.
    """
    query = (query or "").strip()
    if not query: return ("⚠️ Bạn hãy nhập nhu cầu + mức giá (ví dụ: 20 triệu).", [])

    # 1. Intent Detection
    # Phát hiện ý định
    auto_purpose = detect_purpose_from_query(query)
    purpose = (purpose or auto_purpose or "office").lower().strip()

    # 2. Price Extraction
    # Trích xuất giá
    low, high, has_price = extract_price_range(query)
    if not has_price: return ("⚠️ Vui lòng nhập rõ **mức giá** (vd: *20 triệu*, *15-20tr*) để hệ thống tư vấn chính xác.", [])

    # Expand Price Range
    # Mở rộng khoảng giá
    delta = max(0, int(expand_million)) * 1_000_000
    low2 = max(0, low - delta)
    high2 = high + delta

    # Brand Detection
    # Phát hiện thương hiệu
    wanted_canons = canonicalize_brands(brands)
    if not wanted_canons:
        detected = detect_brand_from_query(query)
        if detected: wanted_canons = [detected]

    # 3. Vector Search
    # Tìm kiếm vector
    candidates = [enrich_item(x) for x in vector_search(query, top_k=top_k)]

    # 4. Filter by Price & Brand
    # Lọc theo giá và thương hiệu
    filtered = filter_by_price(candidates, low2, high2)
    if wanted_canons:
        filtered = [x for x in filtered if item_matches_brand(x, wanted_canons)]

    # 5. Fallback Scan
    # Quét dự phòng
    if len(filtered) < return_k:
        scanned = scan_db_by_price_brand(low2, high2, wanted_canons, need=return_k * 6)
        seen = set()
        merged = []
        for it in filtered + scanned:
            key = (norm_text(it.get("name", "")), _to_int(it.get("price_value", 0), 0))
            if key in seen: continue
            merged.append(it)
            seen.add(key)
        filtered = merged

    # 6. Advanced Filtering & Reranking
    # Lọc nâng cao và xếp hạng lại
    filtered = filter_by_purpose_strict(filtered, purpose)
    ranked = rerank(filtered, purpose)
    top = ranked[:return_k]

    # 7. AI Generation
    # Tạo lời khuyên AI
    advice = generate_llm_advice(query, purpose, top)
    
    # 8. Log Interaction
    # Ghi log tương tác
    log_user_interaction(query, purpose, (low, high), advice, top)
    
    return advice, top