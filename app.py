import streamlit as st
import json
import html
import os
import textwrap
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# =============================================================================
# 1. CONFIGURATION & ENVIRONMENT SETUP (CẤU HÌNH & THIẾT LẬP MÔI TRƯỜNG)
# =============================================================================
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ADMIN_PASSWORD = "k37tlu"

# Initialize Groq Client / Khởi tạo Groq Client
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    groq_client = None

# Import Backend Core Functions / Import các hàm xử lý từ Backend
try:
    # Sửa thành src.chatbot_rag_core
    from src.chatbot_rag_core import get_answer, detect_purpose_from_query
except ImportError:
    # Dummy function for fallback if backend is missing
    # Hàm giả lập để tránh crash nếu thiếu backend
    def get_answer(q, p, e, b): return "", [] 
    def detect_purpose_from_query(q): return "office"

# =============================================================================
# 2. GOOGLE SHEETS INTEGRATION (KẾT NỐI GOOGLE SHEETS)
# =============================================================================
def connect_to_gsheet():
    """
    Connect to Google Sheets using credentials from Streamlit secrets.
    Kết nối tới Google Sheets sử dụng thông tin bảo mật trong st.secrets.
    """
    if "gcp_service_account" not in st.secrets:
        return None

    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Open the specific sheet by name
        sheet = client.open("Laptop_Bot_Data").sheet1
        return sheet
    except Exception as e:
        # Silent fail to avoid disrupting user experience
        print(f"GSheet Connection Error: {e}")
        return None

def log_user_data(query, purpose, result_count, products):
    """
    Log user interaction and Top 5 recommendations to Google Sheets.
    Ghi lại tương tác người dùng và danh sách Top 5 máy gợi ý vào Google Sheets.
    """
    try:
        sheet = connect_to_gsheet()
        if sheet:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format Top 5 products as a single string block
            # Định dạng Top 5 sản phẩm thành một chuỗi văn bản
            top_products_str = ""
            if products:
                top_5 = products[:5]
                lines = []
                for i, p in enumerate(top_5):
                    price = f"{p.get('price_value', 0):,.0f}"
                    score = round(p.get('smart_score', 0), 1)
                    # Format: #1 Name (Price - Score)
                    lines.append(f"#{i+1} {p.get('name')} ({price}đ - {score}%)")
                top_products_str = "\n".join(lines)
            # Append row: Timestamp | Query | Purpose | Count | Top Results
            sheet.append_row([timestamp, query, purpose, result_count, top_products_str])
            
    except Exception as e:
        print(f"Log Error: {e}")
# =============================================================================
# 3. HELPER FUNCTIONS (CÁC HÀM HỖ TRỢ)
# =============================================================================
def remove_accents(input_str: str) -> str:
    """
    Normalize Vietnamese text by removing accents (Handling 'đ' -> 'd').
    Chuẩn hóa tiếng Việt: loại bỏ dấu và chuyển đổi 'đ' thành 'd'.
    Args:
        input_str (str): Original string with accents / Chuỗi gốc có dấu.
    Returns:
        str: Normalized string (lowercase, no accents) / Chuỗi đã chuẩn hóa (chữ thường, không dấu).
    """
    if not input_str: 
        return ""
    input_str = input_str.replace("đ", "d").replace("Đ", "D")
    # 2. Normalize Unicode characters to NFD form (Decompose characters and diacritics)
    # Chuyển đổi chuỗi sang chuẩn NFD (Tách rời ký tự gốc và dấu câu)
    s1 = unicodedata.normalize('NFD', input_str)
    # 3. Filter out non-spacing mark characters (The accents)
    # Lọc bỏ các ký tự thuộc loại 'Mn' (Mark, Nonspacing - chính là các dấu)
    s2 = ''.join(c for c in s1 if unicodedata.category(c) != 'Mn')
    # 4. Return lowercase result for consistent matching
    # Trả về kết quả chữ thường để đồng bộ việc so sánh
    return s2.lower()
def extract_price_from_query(query):
    """Extract numeric price value from user query string."""
    pattern = r"(\d{1,3}(?:[.,]\d{3})*|\d+)\s*(tr|triệu|m|k|vnd|đ)?"
    matches = re.findall(pattern, query.lower())
    for num, unit in matches:
        clean_num = float(num.replace('.', '').replace(',', ''))
        if unit in ['tr', 'triệu', 'm']: return clean_num * 1_000_000
        if unit in ['k']: return clean_num * 1_000
        if clean_num > 500_000: return clean_num
    return None

def detect_intent_advanced(query):
    """
    Advanced Intent Detection based on keywords.
    Phát hiện ý định người dùng nâng cao dựa trên từ khóa.
    """
    q_norm = remove_accents(query)
    # 1. Gaming
    gaming_keywords = ['game', 'choi', 'lol', 'pubg', 'valorant', 'csgo', 'genshin', 'gta', 'steam', 'gaming', 'fifa', 'choi game']
    if any(k in q_norm for k in gaming_keywords): return "gaming"
    
    # 2. Creator (Design/Edit)
    creator_keywords = ['do hoa', 'thiet ke', 'render', 'dung phim', 'edit', 'video', 'adobe', 'photoshop', 'premiere', '3d', 'cad', 'revit', 'ai', 'kien truc']
    if any(k in q_norm for k in creator_keywords): return "creator"
    
    # 3. Thin & Light (Portability)
    thin_keywords = ['mong', 'nhe', 'di dong', 'doanh nhan', 'macbook', 'air', 'pin trau', 'gon', 'sang chanh']
    if any(k in q_norm for k in thin_keywords): return "thinlight"
    
    # 4. Office (Default)
    office_keywords = ['van phong', 'hoc tap', 'sinh vien', 'word', 'excel', 'powerpoint', 'ke toan']
    if any(k in q_norm for k in office_keywords): return "office"

    return None

def format_storage(val):
    """Format storage capacity (GB/TB) for display."""
    try:
        v = float(val)
        if v <= 4: return f"{int(v)} TB"
        if v >= 1000: return f"{v/1024:.0f} TB"
        return f"{int(v)} GB"
    except: return str(val)

# =============================================================================
# 4. HARDWARE SCORING SYSTEM (HỆ THỐNG CHẤM ĐIỂM PHẦN CỨNG)
# =============================================================================

def get_price_segment(price):
    """Classify product into price segments."""
    if price < 15_000_000: return "Budget"
    if price < 25_000_000: return "Mid"
    if price < 35_000_000: return "Upper-Mid"
    if price < 50_000_000: return "High"
    if price < 80_000_000: return "Premium"
    return "Ultra"

def get_gpu_tier(gpu_str, is_apple, intent):
    """
    Determine GPU Tier (0-10) with updated RTX 5000 Series logic.
    Xác định cấp độ GPU (0-10) - Cập nhật RTX 5000 Series.
    """
    g = gpu_str.upper()
    
    # --- APPLE GPU LOGIC ---
    if is_apple:
        if intent == 'creator':
            if 'MAX' in g: return 8.5
            if 'PRO' in g: return 7.5
            return 6.5 
        # Non-creator logic
        if 'MAX' in g: return 7.0
        if 'PRO' in g: return 6.0
        return 4.5 

    # --- NVIDIA DISCRETE (RTX 5000 SERIES - NEW) ---
    if '5090' in g: return 10.0
    if '5080' in g: return 9.6
    if '5070' in g: return 8.8
    if '5060' in g: return 8.2
    if '5050' in g: return 7.5

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
    
    return 2.0

def get_gpu_expectation(segment):
    """Minimum expected GPU tier for a price segment."""
    if segment == 'Budget': return 3.0
    if segment == 'Mid': return 6.0
    if segment == 'Upper-Mid': return 7.5
    if segment == 'High': return 8.0
    if segment == 'Premium': return 8.5
    if segment == 'Ultra': return 9.5
    return 3.0

def calculate_match_score(product, intent, rank_index):
    """
    Core Scoring Function (0-100). Includes Gaming Desk handling.
    """
    # 1. DATA EXTRACTION
    raw = product.get('raw_source', {})
    if isinstance(raw, str): raw = json.loads(raw) if raw else {}
    
    brand = str(product.get('brand', '')).upper()
    price = float(product.get('price_value', 0))
    gpu_str = str(raw.get('Card màn hình', product.get('gpu', ''))).upper()
    cpu_str = str(raw.get('Công nghệ CPU', product.get('cpu', ''))).upper()
    screen_str = str(raw.get('Màn hình', '')).upper()
    ram = int(product.get('ram_gb', 8))
    ssd_raw = product.get('ssd_gb', 256)
    ssd = float(ssd_raw) if ssd_raw else 256.0
    if ssd < 10: ssd *= 1024
    refresh_rate = int(product.get('refresh_rate_hz', 60))
    weight = float(product.get('weight_kg', 2.0))
    
    # 2. HARDWARE CLASSIFICATION
    segment = get_price_segment(price)
    is_apple = 'APPLE' in brand or any(x in cpu_str for x in ['M1', 'M2', 'M3', 'M4'])
    is_intel_ultra = 'ULTRA' in cpu_str
    
    # Get GPU Tier
    gpu_tier = get_gpu_tier(gpu_str, is_apple, intent)
    
    is_weak_igpu = gpu_tier <= 4.0 and not ('RTX' in gpu_str or 'GTX' in gpu_str or 'RX' in gpu_str)
    is_igpu_general = gpu_tier < 6.0 and not ('RTX' in gpu_str or 'GTX' in gpu_str)
    if is_apple: 
        is_weak_igpu = False
        is_igpu_general = True

    # 🔥 NEW: Logic for Gaming Desk
    # Máy nặng >= 2.3kg vẫn được coi là tốt nếu là Gaming Desk
    is_gaming_desk = intent == 'gaming' and weight >= 2.3
    
    # 3. SCORING COMPONENTS (0-100)
    score_gpu = gpu_tier * 10
    if intent == 'gaming' and is_weak_igpu and not is_apple: score_gpu = 25 
    
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
    
    if intent == 'gaming':
        if refresh_rate >= 144: score_form += 20
        elif refresh_rate < 120 and not is_apple: score_form -= 15
        
    if intent == 'thinlight':
        if weight < 1.3: score_form = 100
        elif weight < 1.5: score_form = 90
        elif weight > 1.8: score_form = 40
        elif weight > 2.2: score_form = 20
    
    # Value Score
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
    
    # 4. WEIGHTED TOTAL (FINAL CALCULATION)
    final_score = 0
    
    if intent == 'gaming':
        # 🔥 Updated Gaming Formula
        final_score = (score_gpu * 0.32) + (score_value * 0.28) + (score_cpu * 0.15) + (score_mem * 0.1) + (score_form * 0.15)
        
        if is_weak_igpu and not is_apple: final_score -= 35
        elif is_igpu_general and not is_apple: final_score -= 15
        if is_apple: final_score -= 35

        # 🔥 Compensation for heavy gaming laptops (Gaming Desk)
        if is_gaming_desk: final_score += 6 

    elif intent == 'creator':
        final_score = (score_value * 0.3) + (score_gpu * 0.25) + (score_cpu * 0.2) + (score_mem * 0.1) + (score_form * 0.15)
        if is_apple: final_score += 5
        if is_weak_igpu and not is_apple: final_score -= 15
        
    elif intent == 'office':
        final_score = (score_value * 0.4) + (score_cpu * 0.2) + (score_mem * 0.2) + (score_form * 0.15) + (score_gpu * 0.05)
        
    elif intent == 'thinlight':
        final_score = (score_form * 0.4) + (score_value * 0.3) + (score_cpu * 0.15) + (score_mem * 0.1) + (score_gpu * 0.05)
        if weight > 2.0: final_score -= 15
        if is_apple: final_score += 5
    
    # 5. RANK DECAY
    if rank_index == 0: decay = 0
    elif rank_index == 1: decay = 0.8
    elif rank_index == 2: decay = 1.6
    else: decay = rank_index * 0.7
    
    final_score = final_score - decay
    min_score = 35.0 if intent == 'gaming' else 50.5
    return min(99.5, max(min_score, final_score))

# =============================================================================
# 5. AI GENERATION (GROQ ADVISOR)
# =============================================================================
def call_groq_analysis(query, intent, top_products):
    """
    Generate professional advice using Groq LLM.
    Tạo lời khuyên chuyên nghiệp sử dụng Groq LLM.
    """
    if not groq_client or not top_products:
        return None

    # Build context from TOP products
    context_lines = []
    for idx, p in enumerate(top_products[:3], start=1):
        raw = p.get('raw_source', {})
        if isinstance(raw, str):
            raw = json.loads(raw) if raw else {}

        context_lines.append(
            f"{idx}. {p.get('name')} | "
            f"Giá: {p.get('price_value', 0):,.0f}đ | "
            f"CPU: {raw.get('Công nghệ CPU', p.get('cpu', ''))} | "
            f"GPU: {raw.get('Card màn hình', p.get('gpu', ''))}"
        )

    context = "\n".join(context_lines)

    prompt = f"""
    Bạn là chuyên gia tư vấn laptop cấp cao, có kinh nghiệm thực tế về phần cứng,
    phân khúc giá và hành vi người dùng.

    Người dùng đang tìm laptop:
    - Nhu cầu: "{query}"
    - Mục đích sử dụng chính: {intent}

    Hệ thống đã sàng lọc và xếp hạng laptop dựa trên:
    - Phân khúc giá so với cấu hình
    - Hiệu năng thực tế theo mục đích sử dụng
    - Mức độ “đáng tiền” (Performance / Price)
    - Trải nghiệm sử dụng dài hạn

    Danh sách TOP laptop phù hợp nhất (đã sắp xếp theo mức độ phù hợp giảm dần):
    {context}

    YÊU CẦU TRẢ LỜI:
    - Viết đúng 5–6 câu, tiếng Việt, giọng tư vấn như nói với khách thật
    - Tập trung phân tích **TOP 1**:
      + Vì sao cấu hình này phù hợp nhất với nhu cầu
      + Vì sao mức giá này là hợp lý trong phân khúc
    - So sánh ngắn gọn với TOP 2 (tối đa 1 câu, chỉ nêu điểm khác biệt chính)
    - Nhấn mạnh yếu tố “đáng tiền” và tình huống sử dụng thực tế
    - Kết luận rõ ràng:
      + Nên chọn TOP 1 trong đa số trường hợp
      + Chỉ nên cân nhắc TOP 2 khi có nhu cầu cụ thể khác

    QUY TẮC BẮT BUỘC:
    - Không nhắc đến AI, thuật toán, điểm số hay hệ thống
    - Không liệt kê thông số dạng bảng
    - Không khen tất cả sản phẩm
    - Không thêm laptop ngoài danh sách
    """

    try:
        chat = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=600
        )
        return chat.choices[0].message.content
    except Exception as e:
        print(f"Groq analysis error: {e}")
        return None

# =============================================================================
# 6. UI LAYOUT & STYLE (GIAO DIỆN & STYLE)
# =============================================================================
st.set_page_config(page_title="AI Laptop Consultant", layout="wide", page_icon="💻")

st.markdown("""
<style>
/* NỀN TỐI CHUNG CHO APP */
.stApp { background: radial-gradient(1200px 600px at 20% 0%, rgba(60,255,160,0.08), transparent 55%), radial-gradient(900px 500px at 90% 10%, rgba(255,90,90,0.08), transparent 60%), #0b0f14; color: #e8eef6; }
.block-container { padding-top: 2.0rem; padding-bottom: 80px; }

/* --- CARD SẢN PHẨM --- */
.card { background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.2); transition: all 0.3s ease; }
.card:hover { border-color: rgba(255,255,255,0.2); transform: translateY(-2px); }

/* Class dành riêng cho Top 1 */
.card.best { border: 2px solid rgba(70,255,170,0.55); background: linear-gradient(180deg, rgba(70,255,170,0.05), rgba(11,15,20,0.8)); box-shadow: 0 0 15px rgba(70,255,170,0.15); }

/* --- CÁC THÀNH PHẦN KHÁC --- */
.name { font-size: 20px; font-weight: 780; letter-spacing: -0.2px; margin-bottom: 8px; color: #fff; }
.price { font-size: 22px; font-weight: 850; color: #ff5a5a; margin-bottom: 12px; }
.badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.badge { display: inline-flex; align-items: center; gap: 8px; padding: 4px 10px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; font-size: 12px; }
.badge b { color: #aaa; font-weight: 600; } .badge span { color: #fff; font-weight: 500; }
.banner { background: rgba(40,160,95,0.15); border-left: 4px solid #46ffaa; padding: 15px; border-radius: 4px; margin: 20px 0; font-size: 16px; line-height: 1.5; }

/* TABLE & FOOTER & HEADER */
.cmp-container { overflow-x: auto; border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; background: rgba(0,0,0,0.3); margin-bottom: 20px; }
table.cmp-table { width: 100%; border-collapse: collapse; font-size: 13px; font-family: sans-serif; }
table.cmp-table th { background: rgba(255,255,255,0.08); color: #46ffaa; padding: 8px 12px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.1); white-space: nowrap; }
table.cmp-table td { padding: 8px 12px; border-bottom: 1px solid rgba(255,255,255,0.05); color: #ddd; vertical-align: top; line-height: 1.4; }
table.cmp-table tr:last-child td { border-bottom: none; }
.footer-fixed { position: fixed; left: 0; bottom: 0; width: 100%; background: rgba(11, 15, 20, 0.95); backdrop-filter: blur(10px); border-top: 1px solid rgba(255,255,255,0.1); padding: 15px; text-align: center; font-size: 13px; color: #888; z-index: 9999; }
.hero-title { font-size: 42px; font-weight: 850; letter-spacing: -0.6px; margin: 0 0 10px 0; background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5, #9d50bb); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
.hero-sub { font-size: 16px; color: rgba(232,238,246,0.65); margin: 0 0 25px 0; text-align: center; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 7. MAIN APPLICATION LOGIC (LOGIC CHÍNH CỦA ỨNG DỤNG)
# =============================================================================

# Header & Admin Layout
col_shield, col_title, col_dummy = st.columns([1, 10, 1])

with col_shield:
    if st.button("🛡️", key="adm_btn", help="Khu vực quản trị"):
        st.session_state.show_admin = not st.session_state.get('show_admin', False)

with col_title:
    st.markdown('<div class="hero-title">AI Laptop Consultant</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Hệ thống AI gợi ý laptop thông minh</div>', unsafe_allow_html=True)

# Admin Dashboard (Google Sheets View)
if st.session_state.get('show_admin', False):
    with st.expander("🔐 Admin Dashboard (Google Sheets)", expanded=True):
        with st.form("admin_form"):
            pwd = st.text_input("Nhập mã truy cập:", type="password")
            submit = st.form_submit_button("Truy cập")
            if submit:
                if pwd == ADMIN_PASSWORD:
                    st.success("Đang kết nối Google Sheets...")            
                    my_sheet_link = st.secrets["general"]["sheet_link"] 
                    st.link_button("📂 Mở file Google Sheet gốc", my_sheet_link)
                    sheet = connect_to_gsheet()
                    if sheet:
                        try:
                            data = sheet.get_all_records()
                            if data:
                                st.write(f"📊 Tìm thấy {len(data)} bản ghi:")
                                st.dataframe(data[::-1], use_container_width=True)
                            else:
                                st.warning("Sheet trống.")
                        except Exception as e:
                            st.error(f"Lỗi đọc dữ liệu: {e}")
                    else:
                        st.error("Không kết nối được Google Sheet. Kiểm tra secrets.")
                else:
                    st.error("Sai mật khẩu!")

# Auto-update Purpose Callback
def auto_update_purpose():
    if "query_input" in st.session_state:
        q = st.session_state.query_input
        new_intent = detect_intent_advanced(q)
        if new_intent:
            st.session_state.purpose_select = new_intent

# Input Form
c1, c2 = st.columns([2, 1])
with c1:
    query = st.text_input("💬 Nhu cầu & Ngân sách", 
                          placeholder="VD: Laptop gaming 20 triệu, macbook làm mỏng nhẹ...", 
                          key="query_input", 
                          on_change=auto_update_purpose)
with c2:
    purpose = st.selectbox("🎯 Mục đích chính", ["office", "gaming", "creator", "thinlight"], 
                           format_func=lambda x: {"office": "Văn phòng / Học tập", "gaming": "Chơi Game", "creator": "Đồ họa / Kỹ thuật", "thinlight": "Mỏng nhẹ"}[x], 
                           key="purpose_select")

with st.expander("⚙️ Bộ lọc nâng cao"):
    f1, f2 = st.columns(2)
    with f1: brands = st.multiselect("🏷️ Hãng", ["Asus", "Acer", "Dell", "HP", "Lenovo", "MSI", "Apple", "Gigabyte"])
    with f2: expand = st.slider("📈 Biên độ giá (± tr)", 0, 5, 2)

# Action & Results
if st.button("🔍 Tìm kiếm & Tư vấn ngay", type="primary", use_container_width=True):
    if not query.strip():
        st.toast("⚠️ Vui lòng nhập nội dung!", icon="❌")
    else:
        detected_price = extract_price_from_query(query)
        if not detected_price:
             st.toast("⚠️ Vui lòng nhập mức ngân sách (VD: 30tr, 50 triệu)!", icon="💰")
        else:
            with st.spinner("🤖 AI đang phân tích phần cứng & giá..."):
                core_ans, products = get_answer(query, purpose, expand, brands)
                
                if products:
                    # 1. Calculate Score (First Pass)
                    for p in products:
                        p['smart_score'] = calculate_match_score(p, purpose, 0)
                    
                    # 2. Sort by Score (Descending)
                    products.sort(key=lambda x: x['smart_score'], reverse=True)
                    
                    # 3. Recalculate with Rank Decay
                    for i, p in enumerate(products):
                        p['smart_score'] = calculate_match_score(p, purpose, i)

                    # 4. Log User Data (After Sorting)
                    log_user_data(query, purpose, len(products), products)

                    # 5. Generate AI Advice
                    groq_advice = call_groq_analysis(query, purpose, products)
                    final_ans = groq_advice if groq_advice else core_ans
                else:
                    final_ans = "Không tìm thấy sản phẩm nào trong tầm giá này."
                    log_user_data(query, purpose, 0, [])
                
                st.session_state.search_results = (final_ans, products)

ans, products = st.session_state.get("search_results", (None, []))

# =============================================================================
# 8. RESULTS DISPLAY (HIỂN THỊ KẾT QUẢ)
# =============================================================================
if products:
    if ans:
        st.markdown(f'<div class="banner"><b>🤖 AI Advisor:</b> {ans}</div>', unsafe_allow_html=True)

    # Comparison Table
    st.markdown("---")
    product_map = {p['name']: p for p in products}
    selected_names = st.multiselect("⚖️ Chọn 2 máy để so sánh chi tiết:", options=product_map.keys(), max_selections=2)

    if len(selected_names) == 2:
        with st.expander("📊 BẢNG SO SÁNH CHI TIẾT", expanded=True):
            p1, p2 = product_map[selected_names[0]], product_map[selected_names[1]]
            def get_raw(p):
                r = p.get("raw_source", {})
                return json.loads(r) if isinstance(r, str) else (r if r else {})
            
            r1, r2 = get_raw(p1), get_raw(p2)
            all_keys = sorted(list(set(r1.keys()) | set(r2.keys())))
            priority = ["Công nghệ CPU", "RAM", "Ổ cứng", "Card màn hình", "Màn hình", "Tần số quét", "Pin", "Trọng lượng"]
            sorted_keys = [k for k in priority if k in all_keys] + [k for k in all_keys if k not in priority and k != "Tên sản phẩm"]

            rows_html = ""
            for k in sorted_keys:
                v1 = str(r1.get(k, "-"))
                v2 = str(r2.get(k, "-"))
                rows_html += f"<tr><td class='cmp-row-label'>{k}</td><td>{v1}</td><td>{v2}</td></tr>"

            table_html = textwrap.dedent(f"""
                <div class="cmp-container">
                    <table class="cmp-table">
                        <thead>
                            <tr>
                                <th style="width:20%">Thông số</th>
                                <th>{p1['name']}</th>
                                <th>{p2['name']}</th>
                            </tr>
                        </thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>
            """)
            st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"📋 Top {len(products)} Laptop phù hợp nhất")
    
    for i, p in enumerate(products):
        raw = p.get("raw_source", {})
        if isinstance(raw, str): raw = json.loads(raw) if raw else {}
        
        def esc(x): return html.escape(str(x)) if x else "N/A"
        price = p.get("price_value", 0)
        price_str = f"{price:,.0f} VNĐ" if price > 0 else "Liên hệ"
        score = p.get('smart_score', 80)
        score_display = round(score, 1)
        ssd_display = format_storage(p.get("ssd_gb", 0))

        # --- RANK HIGHLIGHT LOGIC ---
        if i == 0:
            card_cls = "best"
            badge_color = "#46ffaa"
        else:
            card_cls = "" 
            badge_color = "#999"

        card_html = textwrap.dedent(f"""
            <div class="card {card_cls}">
                <div class="name">{esc(p.get('name'))}</div>
                <div class="price">{price_str}</div>
                <div class="badges">
                    <div class="badge"><b>CPU</b><span>{esc(p.get("cpu"))}</span></div>
                    <div class="badge"><b>RAM</b><span>{p.get("ram_gb")} GB</span></div>
                    <div class="badge"><b>SSD</b><span>{ssd_display}</span></div>
                    <div class="badge"><b>GPU</b><span>{esc(p.get("gpu"))}</span></div>
                    <div class="badge"><b>Màn</b><span>{esc(p.get("screen_size_inch"))}"</span></div>
                    <div class="badge"><b>Nặng</b><span>{p.get("weight_kg")} kg</span></div>
                    <div class="badge" style="border-color:{badge_color}; color:{badge_color}"><b>Match</b><span>{score_display}%</span></div>
                </div>
            </div>
        """)
        st.markdown(card_html, unsafe_allow_html=True)
        
        with st.expander(f"📄 Chi tiết: {esc(p.get('name'))}"):
            if raw:
                c1, c2 = st.columns(2)
                items = list(raw.items())
                mid = (len(items)+1)//2
                with c1: 
                    for k,v in items[:mid]: st.markdown(f"**{k}:** {v}")
                with c2: 
                    for k,v in items[mid:]: st.markdown(f"**{k}:** {v}")

# =============================================================================
# 9. FOOTER
# =============================================================================
st.markdown("<br><br>", unsafe_allow_html=True)
footer_html = """
<div class="footer-fixed">
    AI Chatbot Project &copy; 2026 &mdash; Data Source: <a href="https://www.thegioididong.com/" target="_blank">Thế Giới Di Động</a>
    <br><i>Note: Prices and promotions are subject to change. / Lưu ý: Giá và khuyến mãi có thể thay đổi.</i>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)