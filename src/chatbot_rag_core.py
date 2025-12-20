import os
import json
import re
import math
import unicodedata
import html
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from chromadb import PersistentClient
import google.generativeai as genai
from groq import Groq 
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & CONSTANTS (CẤU HÌNH & HẰNG SỐ)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent 
DATA_DIR = ROOT / "data"
DB_DIR = DATA_DIR / "chroma_db"
LOG_FILE = DATA_DIR / "user_logs.jsonl" 

COLLECTION_NAME = "laptops"
EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768

DEFAULT_TOP_K = 250   # Retrieve more candidates initially / Lấy nhiều ứng viên ban đầu
DEFAULT_RETURN_K = 5  # Return top 5 to user / Trả về 5 kết quả tốt nhất
SCAN_BATCH = 5000

# -----------------------------------------------------------------------------
# 2. ENVIRONMENT INITIALIZATION (KHỞI TẠO MÔI TRƯỜNG)
# -----------------------------------------------------------------------------
load_dotenv(ROOT / ".env")

# Configure Google AI (For Embeddings)
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Configure Groq AI (For Chat Completion)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Ensure data directory exists / Đảm bảo thư mục data tồn tại
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 3. TEXT UTILITIES (TIỆN ÍCH XỬ LÝ VĂN BẢN)
# -----------------------------------------------------------------------------
def _strip_accents(s: str) -> str:
    """Removes Vietnamese accents and standardizes Unicode."""
    if not s: return ""
    s = s.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s

def norm_text(s: str) -> str:
    """Standardizes input text: lowercase, remove accents, clean spaces."""
    s = (s or "").strip().lower()
    s = _strip_accents(s)
    s = re.sub(r"\s+", " ", s)
    return s

# -----------------------------------------------------------------------------
# 4. INTENT & ENTITY EXTRACTION (TRÍCH XUẤT Ý ĐỊNH & THỰC THỂ)
# -----------------------------------------------------------------------------
PURPOSE_KEYS = {
    "gaming": ["gaming", "choi game", "fps", "rtx", "gtx", "do hoa manh", "game", "chien game"],
    "office": ["van phong", "hoc tap", "excel", "word", "sinh vien", "ke toan", "giao vien", "lam viec"],
    "creator": ["do hoa", "thiet ke", "edit", "dung phim", "premiere", "photoshop", "ai", "render", "video"],
    "thinlight": ["mong nhe", "nhe", "di chuyen", "mang di hoc", "ultrabook", "nho gon", "linh hoat"],
}

def detect_purpose_from_query(query: str) -> Optional[str]:
    """Detects user intent based on keywords."""
    t = norm_text(query)
    for p, keys in PURPOSE_KEYS.items():
        if any(k in t for k in keys):
            return p
    return None

def extract_price_range(query: str) -> Tuple[int, int, bool]:
    """Extracts budget constraints. Returns (min, max, has_price)."""
    t = norm_text(query).replace(",", ".")
    
    # Range (e.g., "15-20")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|den|toi)\s*(\d+(?:\.\d+)?)\s*(?:tr|trieu|m|million)?\b", t)
    if m:
        low, high = float(m.group(1)), float(m.group(2))
        return int(min(low, high) * 1_000_000), int(max(low, high) * 1_000_000), True

    # Single Price (e.g., "25tr") -> +/- 1M range assumption
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(tr|trieu|m|million)\b", t)
    if m:
        x = float(m.group(1))
        return int(max(0, x - 1) * 1_000_000), int((x + 1) * 1_000_000), True

    m = re.search(r"\b(\d{1,3}(?:\.\d+)?)tr\b", t)
    if m:
        x = float(m.group(1))
        return int(max(0, x - 1) * 1_000_000), int((x + 1) * 1_000_000), True

    return 0, 100_000_000, False

# -----------------------------------------------------------------------------
# 5. BRAND HANDLING (XỬ LÝ THƯƠNG HIỆU)
# -----------------------------------------------------------------------------
BRAND_SYNONYMS = {
    "asus": ["asus"], "acer": ["acer"], "dell": ["dell"], "hp": ["hp", "hewlett packard"],
    "lenovo": ["lenovo"], "msi": ["msi"], "apple": ["apple", "macbook", "mac"], "gigabyte": ["gigabyte", "giga"],
}

def canonicalize_brands(brands_ui: Optional[List[str]]) -> List[str]:
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
    t = norm_text(query)
    for brand, keys in BRAND_SYNONYMS.items():
        if any(k in t for k in keys): return brand
    return None

def item_matches_brand(item: Dict[str, Any], wanted_canons: List[str]) -> bool:
    if not wanted_canons: return True
    b_norm = norm_text(str(item.get("brand", "")))
    name_norm = norm_text(str(item.get("name", "")))
    
    # Check raw source for hidden brand names
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

# -----------------------------------------------------------------------------
# 6. VECTOR SEARCH & EMBEDDING (TÌM KIẾM VECTOR)
# -----------------------------------------------------------------------------
def safe_embed(text: str, task_type: str) -> List[float]:
    """Generates embeddings with retry logic."""
    for _ in range(3):
        try:
            resp = genai.embed_content(model=EMBED_MODEL, content=text, task_type=task_type)
            emb = resp["embedding"]
            # Pad or truncate if dimension mismatch
            if len(emb) < EMBED_DIM: emb += [0.0] * (EMBED_DIM - len(emb))
            elif len(emb) > EMBED_DIM: emb = emb[:EMBED_DIM]
            return emb
        except Exception: pass
    return [0.0] * EMBED_DIM

_client = PersistentClient(path=str(DB_DIR))
try: _collection = _client.get_collection(COLLECTION_NAME)
except Exception: _collection = None

def vector_search(query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
    if _collection is None: return []
    emb = safe_embed(query, task_type="retrieval_query")
    try:
        res = _collection.query(query_embeddings=[emb], n_results=top_k)
        return res.get("metadatas", [[]])[0] or []
    except Exception: return []

# -----------------------------------------------------------------------------
# 7. DATA PARSING & ENRICHMENT (PHÂN TÍCH & LÀM GIÀU DỮ LIỆU)
# -----------------------------------------------------------------------------
def _to_int(x, default=0) -> int:
    try: return int(float(str(x).strip()))
    except: return default

def _to_float(x, default=0.0) -> float:
    try: return float(str(x).strip())
    except: return default

def _parse_gb_from_text(s: str) -> int:
    if not s: return 0
    t = norm_text(s).replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*tb\b", t)
    if m: return int(float(m.group(1)) * 1024)
    m = re.search(r"(\d+(?:\.\d+)?)\s*gb\b", t)
    if m: return int(float(m.group(1)))
    return 0

def _parse_screen_inch(s: str) -> float:
    if not s: return 0.0
    m = re.search(r"(\d+(?:\.\d+)?)", str(s))
    return float(m.group(1)) if m else 0.0

def _parse_hz(s: str) -> int:
    if not s: return 0
    m = re.search(r"(\d+)\s*hz", str(s).lower())
    return int(m.group(1)) if m else 0

def _parse_weight_kg(raw: Dict[str, Any]) -> float:
    for k in ["Kích thước", "Khối lượng", "Trong luong"]:
        s = str(raw.get(k, "") or "").lower()
        m = re.search(r"(\d+(?:\.\d+)?)\s*kg", s)
        if m: return float(m.group(1))
    return 0.0

# -----------------------------------------------------------------------------
# 8. ADVANCED HARDWARE LOGIC (LOGIC PHẦN CỨNG CHUYÊN SÂU) - [IMPORTANT]
# Unified with UI Logic (hợp nhất với logic App.py)
# -----------------------------------------------------------------------------
def get_price_segment(price):
    if price < 15_000_000: return "Budget"
    if price < 25_000_000: return "Mid"
    if price < 35_000_000: return "Upper-Mid"
    if price < 50_000_000: return "High"
    if price < 80_000_000: return "Premium"
    return "Ultra"

def get_gpu_expectation(segment):
    """Minimum expected GPU tier for value calculation"""
    if segment == 'Budget': return 3.0
    if segment == 'Mid': return 6.0 
    if segment == 'Upper-Mid': return 7.5
    if segment == 'High': return 8.0 
    if segment == 'Premium': return 8.5 
    if segment == 'Ultra': return 9.5
    return 3.0

# =============================================================================
# 1. GPU CLASSIFICATION SYSTEM (HỆ THỐNG PHÂN LOẠI GPU)
# =============================================================================
def get_gpu_tier(gpu_str: str, is_apple: bool, intent: str) -> float:
    """
    Determine GPU Tier (0-10) with updated RTX 5000 Series logic.
    Xác định cấp độ GPU (0-10) - Cập nhật logic RTX 5000 Series mới nhất.
    """
    g = str(gpu_str).upper()
    
    # --- APPLE GPU LOGIC ---
    if is_apple:
        if intent == 'creator':
            if 'MAX' in g: return 8.5
            if 'PRO' in g: return 7.5
            return 6.5 
        # Non-creator logic / Logic cho nhu cầu thường
        if 'MAX' in g: return 7.0
        if 'PRO' in g: return 6.0
        return 4.5 

    # --- NVIDIA DISCRETE (RTX 5000 SERIES - NEW) ---
    # Logic for new Gen 5000: Higher tier than 4000 equivalent
    if '5090' in g: return 10.0
    if '5080' in g: return 9.6
    if '5070' in g: return 8.8  # Stronger than 4070 (8.5)
    if '5060' in g: return 8.2  # Stronger than 4060 (7.5)
    if '5050' in g: return 7.5  # Stronger than 4050 (7.0)

    # --- RTX 4000 SERIES ---
    if '4090' in g: return 9.8
    if '4080' in g: return 9.3
    if '4070' in g: return 8.5
    if '4060' in g: return 7.5
    if '4050' in g: return 7.0
    
    # --- OLDER SERIES ---
    if '3080' in g: return 8.5
    if '3070' in g: return 7.5
    if '3060' in g: return 6.5
    if '3050' in g: return 6.0
    if '2050' in g or 'GTX' in g or '1650' in g: return 5.0
    
    # --- AMD & iGPU ---
    if 'RX 7600' in g or 'RX 6700' in g: return 7.0
    if 'RX 6600' in g: return 6.5
    if 'ARC' in g: return 5.5 if intent != 'gaming' else 4.5
    if '780M' in g or '680M' in g or '880M' in g or '890M' in g: return 6.0
    if 'ULTRA' in g: return 5.0
    if 'VEGA' in g or 'RADEON' in g: return 4.0
    if 'IRIS' in g or 'XE' in g or 'UHD' in g: return 3.0
    
    return 2.0 # Fallback / Mặc định thấp nhất

# =============================================================================
# 2. DATA ENRICHMENT (LÀM GIÀU DỮ LIỆU)
# =============================================================================
def enrich_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes product data fields from raw source.
    Chuẩn hóa các trường dữ liệu sản phẩm từ nguồn thô.
    """
    out = dict(item)
    raw = out.get("raw_source", {})
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except: raw = {}
    out["raw_source"] = raw

    # 1. Normalize Name & Brand / Chuẩn hóa Tên & Hãng
    if not out.get("name"): out["name"] = raw.get("Tên sản phẩm", "")
    if not out.get("brand"):
        n = norm_text(out.get("name", ""))
        for canon, keys in BRAND_SYNONYMS.items():
            if any(k in n for k in keys):
                out["brand"] = canon
                break
    else: out["brand"] = norm_text(str(out["brand"]))

    # 2. Parse Price / Xử lý Giá
    out["price_value"] = _to_int(out.get("price_value") or raw.get("Giá", 0))

    # 3. Normalize Hardware Specs / Chuẩn hóa thông số phần cứng
    if not out.get("cpu"): out["cpu"] = raw.get("Công nghệ CPU", "")
    if not out.get("gpu"): out["gpu"] = raw.get("Card màn hình", "")

    if _to_int(out.get("ram_gb")) <= 0: out["ram_gb"] = _parse_gb_from_text(str(raw.get("RAM", "")))
    if _to_int(out.get("ssd_gb")) <= 0: out["ssd_gb"] = _parse_gb_from_text(str(raw.get("Ổ cứng", "")))
    if _to_float(out.get("screen_size_inch")) <= 0: out["screen_size_inch"] = _parse_screen_inch(str(raw.get("Kích thước màn hình", "")))
    if _to_int(out.get("refresh_rate_hz")) <= 0: out["refresh_rate_hz"] = _parse_hz(str(raw.get("Tần số quét", "")))
    if _to_float(out.get("weight_kg")) <= 0: out["weight_kg"] = _parse_weight_kg(raw)

    # 4. Screen Analysis / Phân tích màn hình
    raw_screen = norm_text(str(raw.get("Công nghệ màn hình", "") + raw.get("Màn hình", "") + raw.get("Độ phân giải", "")))
    out["is_ips"] = "ips" in raw_screen
    out["is_oled"] = "oled" in raw_screen
    out["is_high_res"] = any(x in raw_screen for x in ["2k", "4k", "qhd", "retina", "2.8k", "3k"])
    out["features"] = { "is_ips": out["is_ips"], "is_oled": out["is_oled"], "high_res": out["is_high_res"] }

    return out

# =============================================================================
# 3. FILTERING LOGIC (LOGIC LỌC SẢN PHẨM)
# =============================================================================
def filter_by_price(items: List[Dict[str, Any]], low: int, high: int) -> List[Dict[str, Any]]:
    """Filter products within price range."""
    out = []
    for it in items:
        price = _to_int(it.get("price_value", 0), 0)
        if low <= price <= high:
            out.append(it)
    return out

def filter_by_purpose_strict(items: List[Dict[str, Any]], purpose: str) -> List[Dict[str, Any]]:
    """
    Relaxed filtering: Avoid hard-rejecting gaming laptops for office use.
    Logic lọc nới lỏng hơn để không loại bỏ oan máy Gaming khi dùng Office.
    """
    p = (purpose or "").lower().strip()
    if not p: return items

    out = []
    for it in items:
        gpu = norm_text(it.get("gpu", ""))
        w = _to_float(it.get("weight_kg", 0.0), 0.0)
        
        has_dgpu = any(x in gpu for x in ["rtx", "gtx", "card roi", "nvidia", "amd radeon"])
        
        # Gaming: Needs Discrete GPU / Cần Card rời
        if p == "gaming":
            if has_dgpu or "game" in norm_text(it.get("name","")): out.append(it)
        
        # Office: Gaming laptops are okay for office / Gaming làm văn phòng vẫn tốt
        elif p == "office":
            out.append(it) 
        
        # Thinlight: Filter by weight / Lọc theo trọng lượng (Ưu tiên < 1.9kg)
        elif p == "thinlight":
            if w > 0 and w <= 1.9: out.append(it)
            elif w == 0: out.append(it) # Keep unknown weight for scoring
        
        # Creator: Needs decent RAM / Cần RAM ổn
        elif p == "creator":
            if _to_int(it.get("ram_gb", 0)) >= 8: out.append(it)
        
        else:
            out.append(it)
    return out

# =============================================================================
# 4. SCORING SYSTEM (HỆ THỐNG CHẤM ĐIỂM)
# =============================================================================
def calculate_score(item: Dict, purpose: str, center_price: int = 0) -> float:
    """
    Unified Scoring Logic (Logic chấm điểm hợp nhất).
    Matches the logic in Streamlit UI (app.py) EXACTLY.
    """
    # 1. DATA EXTRACTION
    raw = item.get("raw_source", {})
    if isinstance(raw, str): raw = json.loads(raw)
    
    brand = str(item.get('brand', '')).upper()
    price = float(item.get('price_value', 0))
    gpu_str = str(item.get('gpu', '')).upper()
    cpu_str = str(item.get('cpu', '')).upper()
    screen_str = str(raw.get('Màn hình', '')).upper()
    ram = int(item.get('ram_gb', 8))
    ssd = int(item.get('ssd_gb', 256))
    refresh_rate = int(item.get('refresh_rate_hz', 60))
    weight = float(item.get('weight_kg', 2.0))

    # 2. HARDWARE CLASSIFICATION
    segment = get_price_segment(price)
    is_apple = 'APPLE' in brand or any(x in cpu_str for x in ['M1', 'M2', 'M3', 'M4'])
    is_intel_ultra = 'ULTRA' in cpu_str
    
    # Get GPU Tier using the Advanced Logic
    gpu_tier = get_gpu_tier(gpu_str, is_apple, purpose)
    
    # Identify iGPU types
    is_weak_igpu = gpu_tier <= 4.0 and not ('RTX' in gpu_str or 'GTX' in gpu_str or 'RX' in gpu_str)
    is_igpu_general = gpu_tier < 6.0 and not ('RTX' in gpu_str or 'GTX' in gpu_str)
    if is_apple:
        is_weak_igpu = False
        is_igpu_general = True

    # 🔥 NEW: GAMING DESK LOGIC (SYNCED WITH APP.PY)
    # Máy nặng >= 2.3kg vẫn được coi là tốt nếu là Gaming Desk
    is_gaming_desk = purpose == 'gaming' and weight >= 2.3

    # 3. COMPONENT SCORING (0-100)
    
    # --- 3.1 GPU ---
    score_gpu = gpu_tier * 10
    if purpose == 'gaming' and is_weak_igpu and not is_apple: score_gpu = 25

    # --- 3.2 CPU ---
    score_cpu = 60
    if is_apple: score_cpu = 90
    elif is_intel_ultra: score_cpu = 85
    elif 'I9' in cpu_str or 'R9' in cpu_str: score_cpu = 90
    elif 'I7' in cpu_str or 'R7' in cpu_str: score_cpu = 80
    elif 'I5' in cpu_str or 'R5' in cpu_str: score_cpu = 70
    elif 'I3' in cpu_str: score_cpu = 50

    # --- 3.3 RAM/SSD ---
    score_mem = 80
    if ram < 8: score_mem = 30
    elif ram >= 16: score_mem = 100
    if price > 20_000_000 and ssd < 512: score_mem -= 20

    # --- 3.4 FORM/SCREEN ---
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

    # --- 3.5 VALUE SCORE ---
    score_value = 80
    expected_gpu = get_gpu_expectation(segment)
    
    # Penalty for bad specs in high tiers / Trừ điểm nếu cấu hình thấp ở phân khúc cao
    if ram < 16 and segment in ['High', 'Premium', 'Ultra']: score_value -= 15
    if ssd < 512 and segment != 'Budget': score_value -= 10

    if is_apple:
        if segment == 'Ultra' and 'MAX' not in cpu_str: score_value -= 20
        elif segment == 'Premium' and 'PRO' not in cpu_str: score_value -= 10
    else:
        diff = gpu_tier - expected_gpu
        if diff >= 1.0: score_value += 15 
        elif diff <= -2.0: score_value -= 40 # Severe penalty / Phạt nặng
        elif diff <= -1.0: score_value -= 25

    # 4. WEIGHTED SUM (UPDATED FORMULA)
    final_score = 0
    if purpose == 'gaming':
        # 🔥 Updated Formula (New weights & Desk compensation)
        final_score = (score_gpu * 0.32) + (score_value * 0.28) + (score_cpu * 0.15) + (score_mem * 0.1) + (score_form * 0.15)
        
        if is_weak_igpu and not is_apple: final_score -= 35
        elif is_igpu_general and not is_apple: final_score -= 15
        if is_apple: final_score -= 35

        # 🔥 Compensation for Gaming Desk / Bù điểm cho máy nặng
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

    return float(max(0.0, min(100.0, final_score)))

def rerank(items: List[Dict[str, Any]], purpose: str, center_price: int) -> List[Dict[str, Any]]:
    for it in items:
        # Use the unified calculate_score function
        it["fit_score"] = round(calculate_score(it, purpose, center_price), 1)
    
    def key(it: Dict[str, Any]):
        return -it.get("fit_score", 0.0) # Sort by score DESC
    
    return sorted(items, key=key)

# -----------------------------------------------------------------------------
# 10. LLM GENERATION (SINH LỜI KHUYÊN TỪ AI)
# -----------------------------------------------------------------------------
def generate_llm_advice(query: str, purpose: str, products: List[Dict[str, Any]]) -> str:
    if not products:
        return "Mình chưa tìm được mẫu laptop phù hợp trong tầm giá này. Bạn thử tăng ngân sách hoặc đổi nhu cầu nhé."

    lines = []
    for i, p in enumerate(products[:3], 1):
        line = (
            f"{i}. {p.get('name')} - Giá: {p.get('price_value'):,} VNĐ\n"
            f"   - CPU: {p.get('cpu')}, GPU: {p.get('gpu')}\n"
            f"   - RAM: {p.get('ram_gb')}GB, SSD: {p.get('ssd_gb')}GB\n"
        )
        lines.append(line)
    
    product_text = "\n".join(lines)
    prompt = f"""
Bạn là chuyên gia tư vấn laptop chuyên nghiệp.
Khách hàng đang tìm: "{query}" (Nhu cầu: {purpose})

Top 3 laptop tốt nhất hệ thống tìm được:
{product_text}

YÊU CẦU:
1. Viết đoạn tư vấn ngắn (3-4 câu) tiếng Việt.
2. Khuyên chọn máy nào tốt nhất và giải thích lý do (cấu hình/giá).
3. Giọng văn tự nhiên, hữu ích.
    """

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Bạn là trợ lý ảo tư vấn laptop chuyên nghiệp."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Groq API Error: {e}")
        return "Hệ thống đã tìm thấy các sản phẩm phù hợp nhất. Mời bạn xem chi tiết bên dưới."

def scan_db_by_price_brand(low: int, high: int, wanted_canons: List[str], need: int) -> List[Dict[str, Any]]:
    """Fallback: Linear scan if vector search fails."""
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

# -----------------------------------------------------------------------------
# 11. LOGGING (GHI LOG)
# -----------------------------------------------------------------------------
def log_user_interaction(query: str, purpose: str, price_range: Tuple[int, int], advice: str, products: List[Dict]):
    try:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "intent": purpose,
            "price_min": price_range[0],
            "price_max": price_range[1],
            # Truncate advice to save space / Cắt ngắn advice để tiết kiệm dung lượng log
            "ai_response": advice[:200] + "..." if len(advice) > 200 else advice,
            "recommended_models": [p.get("name", "Unknown") for p in products[:5]]
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"⚠️ Logger Error: {e}")

# -----------------------------------------------------------------------------
# 12. MAIN PUBLIC INTERFACE (GIAO DIỆN CHÍNH)
# -----------------------------------------------------------------------------
def get_answer(
    query: str,
    purpose: str,
    expand_million: int,
    brands: Optional[List[str]] = None,
    top_k: int = DEFAULT_TOP_K,
    return_k: int = DEFAULT_RETURN_K,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Orchestrates the entire RAG pipeline."""
    query = (query or "").strip()
    if not query: return ("⚠️ Bạn hãy nhập nhu cầu + mức giá (ví dụ: 20 triệu).", [])

    # 1. Intent Detection
    auto_purpose = detect_purpose_from_query(query)
    purpose = (purpose or auto_purpose or "office").lower().strip()

    # 2. Price Extraction
    low, high, has_price = extract_price_range(query)
    if not has_price: return ("⚠️ Vui lòng nhập rõ **mức giá** (vd: *20 triệu*, *15-20tr*) để hệ thống tư vấn chính xác.", [])

    # Expand Price Range
    delta = max(0, int(expand_million)) * 1_000_000
    low2 = max(0, low - delta)
    high2 = high + delta
    center = (low + high) // 2

    # Brand Detection
    wanted_canons = canonicalize_brands(brands)
    if not wanted_canons:
        detected = detect_brand_from_query(query)
        if detected: wanted_canons = [detected]

    # 3. Vector Search
    candidates = [enrich_item(x) for x in vector_search(query, top_k=top_k)]

    # 4. Filter by Price & Brand
    filtered = filter_by_price(candidates, low2, high2)
    if wanted_canons:
        filtered = [x for x in filtered if item_matches_brand(x, wanted_canons)]

    # 5. Fallback Scan
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

    # 6. Advanced Filtering & Context-Aware Reranking (UNIFIED LOGIC)
    filtered = filter_by_purpose_strict(filtered, purpose)
    ranked = rerank(filtered, purpose, center)
    top = ranked[:return_k]

    # 7. AI Generation
    advice = generate_llm_advice(query, purpose, top)
    
    # 8. Log Interaction
    log_user_interaction(query, purpose, (low, high), advice, top)
    
    return advice, top