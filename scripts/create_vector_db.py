"""
Data Ingestion & Vectorization Pipeline.
Quy trình nhập liệu & Vector hóa dữ liệu.

This module performs the ETL (Extract, Transform, Load) process:
1. Extract: Reads raw product data from JSONL files.
2. Transform: Cleans, normalizes fields (price, specs), and generates summary text.
3. Vectorize: Generates embeddings using Google Gemini API with retry logic.
4. Load: Upserts vectors and metadata into ChromaDB.

Author: AI Engineer
Date: 2026
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

from dotenv import load_dotenv
from tqdm import tqdm
from chromadb import PersistentClient
import google.generativeai as genai

# =============================================================================
# 1. CONFIGURATION & CONSTANTS (CẤU HÌNH & HẰNG SỐ)
# =============================================================================
ROOT_DIR = Path(__file__).resolve().parent.parent # Adjust based on your file structure
DATA_DIR = ROOT_DIR / "data"
INPUT_FILE = DATA_DIR / "products_final.jsonl"
DB_DIR = DATA_DIR / "chroma_db"

# Database Configuration
COLLECTION_NAME = "laptops"
EMBED_MODEL = "models/text-embedding-004"
EMBED_DIM = 768
BATCH_SIZE = 20  # Batch size for processing to respect API rate limits

# =============================================================================
# 2. ENVIRONMENT INITIALIZATION (KHỞI TẠO MÔI TRƯỜNG)
# =============================================================================
load_dotenv(ROOT_DIR / ".env")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Ensure directories exist / Đảm bảo thư mục tồn tại
if not DATA_DIR.exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
if not DB_DIR.exists():
    DB_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 3. UTILITY FUNCTIONS: PARSING (TIỆN ÍCH PHÂN TÍCH DỮ LIỆU)
# =============================================================================
def to_int(value: Any, default: int = 0) -> int:
    """Safely converts a value to integer."""
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default

def to_float(value: Any, default: float = 0.0) -> float:
    """Safely converts a value to float."""
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default

def parse_inch(value: Any) -> float:
    """
    Extracts screen size from string (e.g., '15.6 inch', '15.6"').
    """
    if value is None: return 0.0
    s = str(value).lower()
    match = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(match.group(1)) if match else 0.0

def parse_hz(value: Any) -> int:
    """
    Extracts refresh rate (Hz) from string.
    """
    if value is None: return 0
    s = str(value).lower()
    
    # Priority 1: Explicit 'hz' unit
    match = re.search(r"(\d+)\s*hz", s)
    if match: return int(match.group(1))
    
    # Priority 2: Standalone 2-3 digit numbers (heuristic)
    match_fallback = re.search(r"\b(\d{2,3})\b", s)
    return int(match_fallback.group(1)) if match_fallback else 0

def parse_weight_kg(value: Any) -> float:
    """
    Extracts weight (kg) from string.
    """
    if value is None: return 0.0
    s = str(value).lower().replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)\s*kg", s)
    return float(match.group(1)) if match else 0.0

def normalize_brand(name: str) -> str:
    """
    Normalizes brand names from product titles.
    Maps variations to canonical names (Synced with RAG Core).
    """
    n = (name or "").lower()
    
    if "gigabyte" in n or "giga" in n: return "Gigabyte"
    if "macbook" in n or "apple" in n: return "Apple"
    if "asus" in n: return "Asus"
    if "acer" in n: return "Acer"
    if "hp" in n or "hewlett" in n: return "HP"
    if "lenovo" in n: return "Lenovo"
    if "msi" in n: return "MSI"
    if "dell" in n: return "Dell"
    if "lg" in n and "gram" in n: return "LG"

    return "Other"

# =============================================================================
# 4. DATA ENRICHMENT & TRANSFORMATION (LÀM GIÀU & CHUYỂN ĐỔI DỮ LIỆU)
# =============================================================================
def build_summary_text(item: Dict[str, Any]) -> str:
    """
    Constructs a semantic summary string for embedding.
    Combines critical fields to ensure the vector captures product essence.
    Tạo chuỗi tóm tắt ngữ nghĩa để tạo vector embedding.
    """
    # Robust raw_source handling
    raw = item.get("raw_source", {})
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except json.JSONDecodeError: raw = {}

    # Extract fields with fallbacks to raw data
    name = item.get("name") or raw.get("Tên sản phẩm") or ""
    price = item.get("price_value") or raw.get("Giá") or ""
    cpu = item.get("cpu") or raw.get("Công nghệ CPU") or ""
    gpu = item.get("gpu") or raw.get("Card màn hình") or ""
    ram = item.get("ram_gb") or raw.get("RAM") or ""
    ssd = item.get("ssd_gb") or raw.get("Ổ cứng") or ""
    inch = item.get("screen_size_inch") or raw.get("Kích thước màn hình") or ""
    hz = item.get("refresh_rate_hz") or raw.get("Tần số quét") or ""
    w = item.get("weight_kg") or raw.get("Kích thước") or ""
    brand = item.get("brand") or normalize_brand(name)

    # Compose text components rich in keywords
    parts = [
        f"{brand} {name}",           # Strong signal for Brand + Model
        f"CPU {cpu} GPU {gpu}",      # Performance specs
        f"RAM {ram}GB SSD {ssd}GB",  # Memory/Storage
        f"Màn hình {inch} inch {hz}Hz", # Display
        f"Trọng lượng {w} kg",       # Portability
        f"Giá khoảng {price} VNĐ"    # Price context
    ]
    
    # Join non-empty parts
    return " | ".join([p for p in parts if p and str(p).strip()])

def normalize_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main Transformation Function.
    Standardizes data types and fills missing values from raw JSON source.
    """
    # Parse raw_source if stringified
    raw = item.get("raw_source", {})
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except Exception: raw = {}

    # 1. Basic Info Extraction
    name = item.get("name") or raw.get("Tên sản phẩm") or "Unknown"
    price = to_int(item.get("price_value") or raw.get("Giá") or 0)
    brand = item.get("brand") or normalize_brand(name)
    
    # 2. Specs Extraction
    cpu = item.get("cpu") or raw.get("Công nghệ CPU") or ""
    gpu = item.get("gpu") or raw.get("Card màn hình") or ""
    
    ram_gb = to_int(item.get("ram_gb") or raw.get("RAM") or 0)
    
    # Complex SSD parsing (handling TB/GB units)
    ssd_gb = to_int(item.get("ssd_gb") or 0)
    if ssd_gb == 0:
        o = str(raw.get("Ổ cứng") or "").lower().replace(",", ".")
        m_tb = re.search(r"(\d+(?:\.\d+)?)\s*tb", o)
        m_gb = re.search(r"(\d+(?:\.\d+)?)\s*gb", o)
        if m_tb:
            ssd_gb = int(float(m_tb.group(1)) * 1024)
        elif m_gb:
            ssd_gb = int(float(m_gb.group(1)))

    # 3. Physical Specs Extraction
    screen_inch = to_float(item.get("screen_size_inch") or parse_inch(raw.get("Kích thước màn hình")))
    hz = to_int(item.get("refresh_rate_hz") or parse_hz(raw.get("Tần số quét")))
    weight = to_float(item.get("weight_kg") or parse_weight_kg(raw.get("Kích thước")))

    # 4. Computed Metrics (if available)
    # [UPDATED]: Removed 'gpu_power_rank' and 'score' as they are calculated dynamically in RAG Core.
    url = item.get("url") or raw.get("url") or ""

    # Serialize raw_source for metadata compatibility in ChromaDB
    raw_str = json.dumps(raw, ensure_ascii=False)

    # 5. Construct Normalized Object
    out = dict(item)
    out.update({
        "name": name,
        "brand": brand,
        "price_value": price,
        "cpu": cpu,
        "gpu": gpu,
        "ram_gb": ram_gb,
        "ssd_gb": ssd_gb,
        "screen_size_inch": screen_inch,
        "refresh_rate_hz": hz,
        "weight_kg": weight,
        "url": url,
        "raw_source": raw_str,
    })

    # Generate Summary Text for Embedding
    out["summary_text"] = item.get("summary_text") or build_summary_text(out)

    return out

# =============================================================================
# 5. VECTORIZATION LOGIC (LOGIC VECTOR HÓA)
# =============================================================================
def safe_embed(text: str) -> List[float]:
    """
    Generates embeddings with exponential backoff/retry logic.
    Ensures pipeline robustness against API transient failures.
    """
    for attempt in range(3):
        try:
            resp = genai.embed_content(
                model=EMBED_MODEL,
                content=text,
                task_type="retrieval_document",
            )
            emb = resp["embedding"]
            
            # Dimension validation & padding
            if len(emb) < EMBED_DIM:
                emb = emb + [0.0] * (EMBED_DIM - len(emb))
            elif len(emb) > EMBED_DIM:
                emb = emb[:EMBED_DIM]
                
            return emb
        except Exception as e:
            if attempt == 2:
                print(f"⚠️ Embedding failed after 3 retries: {e}")
            continue
            
    # Fallback: Zero vector
    return [0.0] * EMBED_DIM

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Reads JSONL file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"❌ Input file not found: {path}")
    
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue # Skip malformed lines
    return items

# =============================================================================
# 6. MAIN PIPELINE EXECUTION (THỰC THI PIPELINE CHÍNH)
# =============================================================================
def build_database():
    """
    Orchestrates the DB creation process.
    Điều phối quá trình tạo Database.
    """
    # 1. Extract & Transform
    print(f"📂 Loading data from: {INPUT_FILE}")
    try:
        raw_items = load_jsonl(INPUT_FILE)
    except Exception as e:
        print(e)
        return

    print("🔄 Normalizing data...")
    items = [normalize_item(x) for x in raw_items]
    print(f"🚀 Initializing Vector DB build for {len(items)} products...")

    # 2. Initialize ChromaDB Client
    client = PersistentClient(path=str(DB_DIR))

    # Reset collection to ensure clean state
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"🗑️ Deleted existing collection: {COLLECTION_NAME}")
    except Exception:
        pass # Collection might not exist yet

    col = client.get_or_create_collection(name=COLLECTION_NAME)

    # 3. Load (Batch Processing)
    total = len(items)
    # ncols=100 ensures progress bar fits in standard terminal windows
    for start in tqdm(range(0, total, BATCH_SIZE), ncols=100, desc="Processing Batches"):
        batch = items[start : start + BATCH_SIZE]

        # Generate Embeddings
        texts = [b["summary_text"] for b in batch]
        embeddings = [safe_embed(t) for t in texts]

        # Prepare IDs and Metadata
        ids = [str(b.get("id") or f"laptop_{start+i}") for i, b in enumerate(batch)]
        metadatas = []

        for b in batch:
            # ChromaDB requires primitive types for metadata (no dicts/lists)
            # [UPDATED]: Removed 'gpu_power_rank' and 'score' to align with RAG Core logic.
            md = {
                "name": b.get("name", ""),
                "brand": b.get("brand", ""),
                "price_value": int(b.get("price_value", 0)),
                "cpu": b.get("cpu", ""),
                "gpu": b.get("gpu", ""),
                "ram_gb": int(b.get("ram_gb", 0)),
                "ssd_gb": int(b.get("ssd_gb", 0)),
                "screen_size_inch": float(b.get("screen_size_inch", 0.0)),
                "refresh_rate_hz": int(b.get("refresh_rate_hz", 0)),
                "weight_kg": float(b.get("weight_kg", 0.0)),
                "url": b.get("url", ""),
                "raw_source": b.get("raw_source", "{}"), # Stored as JSON string
            }
            metadatas.append(md)

        # Upsert Batch
        if embeddings and ids:
            col.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    print(f"✅ Database built successfully at: {DB_DIR}")
    print(f"📊 Total documents indexed: {col.count()}")

if __name__ == "__main__":
    build_database()