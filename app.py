import streamlit as st
import json
import html
import os
import textwrap
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# 1. CONFIGURATION & ENVIRONMENT SETUP
env_path = Path(__file__).resolve().parent / '.env'
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ADMIN_PASSWORD = "k37tlu"  # Nên chuyển sang .env hoặc secrets sau

# Initialize Groq Client
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except ImportError:
    groq_client = None

# IMPORT FROM CORE (sau khi đã di chuyển toàn bộ logic vào đây)
try:
    from src.chatbot_rag_core import (
        get_answer,
        detect_purpose_from_query,      # Nếu muốn dùng auto-detect trong UI (tùy chọn)
        extract_price_range             # Dùng để check người dùng có nhập giá chưa
    )
except ImportError:
    # Fallback dummy
    def get_answer(q, p, e, b): return "", []
    def detect_purpose_from_query(q): return "office"
    def extract_price_range(q): return 0, 100_000_000, False

# 2. GOOGLE SHEETS INTEGRATION (giữ nguyên)
def connect_to_gsheet():
    if "gcp_service_account" not in st.secrets:
        return None
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("Laptop_Bot_Data").sheet1
        return sheet
    except Exception as e:
        print(f"GSheet Connection Error: {e}")
        return None

def log_user_data(query, purpose, result_count, products):
    try:
        sheet = connect_to_gsheet()
        if sheet:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            top_products_str = ""
            if products:
                top_5 = products[:5]
                lines = []
                for i, p in enumerate(top_5):
                    price = f"{p.get('price_value', 0):,.0f}"
                    score = round(p.get('fit_score', p.get('smart_score', 0)), 1)  # Dùng fit_score từ core mới
                    lines.append(f"#{i+1} {p.get('name')} ({price}đ - {score}%)")
                top_products_str = "\n".join(lines)
            sheet.append_row([timestamp, query, purpose, result_count, top_products_str])
    except Exception as e:
        print(f"Log Error: {e}")

# 3. HELPER FUNCTION (chỉ giữ lại format_storage)
def format_storage(val):
    try:
        v = float(val)
        if v <= 4: return f"{int(v)} TB"
        if v >= 1000: return f"{v/1024:.0f} TB"
        return f"{int(v)} GB"
    except:
        return str(val)

# 4. AI GENERATION (Groq Advisor - giữ nguyên)
def call_groq_analysis(query, intent, top_products):
    if not groq_client or not top_products:
        return None

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

# 5. UI STYLE (giữ nguyên hoàn toàn)
st.set_page_config(page_title="AI Laptop Consultant", layout="wide", page_icon="💻")

st.markdown("""
<style>
.stApp { background: radial-gradient(1200px 600px at 20% 0%, rgba(60,255,160,0.08), transparent 55%), radial-gradient(900px 500px at 90% 10%, rgba(255,90,90,0.08), transparent 60%), #0b0f14; color: #e8eef6; }
.block-container { padding-top: 2.0rem; padding-bottom: 80px; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 20px; margin: 15px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.2); transition: all 0.3s ease; }
.card:hover { border-color: rgba(255,255,255,0.2); transform: translateY(-2px); }
.card.best { border: 2px solid rgba(70,255,170,0.55); background: linear-gradient(180deg, rgba(70,255,170,0.05), rgba(11,15,20,0.8)); box-shadow: 0 0 15px rgba(70,255,170,0.15); }
.name { font-size: 20px; font-weight: 780; letter-spacing: -0.2px; margin-bottom: 8px; color: #fff; }
.price { font-size: 22px; font-weight: 850; color: #ff5a5a; margin-bottom: 12px; }
.badges { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
.badge { display: inline-flex; align-items: center; gap: 8px; padding: 4px 10px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); border-radius: 6px; font-size: 12px; }
.badge b { color: #aaa; font-weight: 600; } .badge span { color: #fff; font-weight: 500; }
.banner { background: rgba(40,160,95,0.15); border-left: 4px solid #46ffaa; padding: 15px; border-radius: 4px; margin: 20px 0; font-size: 16px; line-height: 1.5; }
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

# 6. MAIN UI & LOGIC
col_shield, col_title, col_dummy = st.columns([1, 10, 1])
with col_shield:
    if st.button("🛡️", key="adm_btn", help="Khu vực quản trị"):
        st.session_state.show_admin = not st.session_state.get('show_admin', False)

with col_title:
    st.markdown('<div class="hero-title">AI Laptop Consultant</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Hệ thống AI gợi ý laptop thông minh</div>', unsafe_allow_html=True)

# Admin Dashboard
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
                        st.error("Không kết nối được Google Sheet.")
                else:
                    st.error("Sai mật khẩu!")

# Input Form
c1, c2 = st.columns([2, 1])
with c1:
    query = st.text_input("💬 Nhu cầu & Ngân sách", 
                          placeholder="VD: Laptop gaming 20 triệu, macbook mỏng nhẹ pin trâu...", 
                          key="query_input")
with c2:
    purpose_options = ["auto", "office", "gaming", "creator", "thinlight"]
    purpose = st.selectbox("🎯 Mục đích chính", purpose_options,
                           format_func=lambda x: {"auto": "Tự động phát hiện", "office": "Văn phòng / Học tập", "gaming": "Chơi Game", "creator": "Đồ họa / Kỹ thuật", "thinlight": "Mỏng nhẹ"}[x],
                           key="purpose_select")

with st.expander("⚙️ Bộ lọc nâng cao"):
    f1, f2 = st.columns(2)
    with f1:
        brands = st.multiselect("🏷️ Hãng", ["Asus", "Acer", "Dell", "HP", "Lenovo", "MSI", "Apple", "Gigabyte"])
    with f2:
        expand = st.slider("📈 Biên độ giá (± tr)", 0, 5, 2)

# Search Action
if st.button("🔍 Tìm kiếm & Tư vấn ngay", type="primary", use_container_width=True):
    if not query.strip():
        st.toast("⚠️ Vui lòng nhập nội dung!", icon="❌")
    else:
        _, _, has_price = extract_price_range(query)
        if not has_price:
            st.toast("⚠️ Vui lòng nhập mức ngân sách (VD: 20tr, dưới 30 triệu, 15-25tr)!", icon="💰")
        else:
            with st.spinner("🤖 AI đang phân tích phần cứng & giá..."):
                # purpose: nếu chọn "auto" thì để None → core sẽ tự detect
                core_purpose = None if purpose == "auto" else purpose
                core_ans, products = get_answer(query, core_purpose, expand, brands)

                if products:
                    # Core đã tự rerank và tính fit_score rồi → chỉ cần log và gọi Groq
                    log_user_data(query, purpose if purpose != "auto" else "auto", len(products), products)

                    groq_advice = call_groq_analysis(query, purpose if purpose != "auto" else "office", products)
                    final_ans = groq_advice if groq_advice else core_ans
                else:
                    final_ans = "Không tìm thấy sản phẩm nào trong tầm giá này."
                    log_user_data(query, purpose if purpose != "auto" else "auto", 0, [])

                st.session_state.search_results = (final_ans, products)

# Display Results
ans, products = st.session_state.get("search_results", (None, []))

if products:
    if ans:
        st.markdown(f'<div class="banner"><b>🤖 AI Advisor:</b> {ans}</div>', unsafe_allow_html=True)

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
        # Dùng fit_score từ core mới
        score = p.get('fit_score', 80)
        score_display = round(score, 1)
        ssd_display = format_storage(p.get("ssd_gb", 0))

        card_cls = "best" if i == 0 else ""
        badge_color = "#46ffaa" if i == 0 else "#999"

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

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
footer_html = """
<div class="footer-fixed">
    AI Chatbot Project &copy; 2026 &mdash; Data Source: <a href="https://www.thegioididong.com/" target="_blank">Thế Giới Di Động</a>
    <br><i>Note: Prices and promotions are subject to change. / Lưu ý: Giá và khuyến mãi có thể thay đổi.</i>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)