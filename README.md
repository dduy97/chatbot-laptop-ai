# 💻 Laptop AI Consultant Chatbot (RAG System)

Hệ thống **tư vấn Laptop thông minh** tích hợp công nghệ  
**RAG – Retrieval-Augmented Generation**.

Ứng dụng kết hợp **LLM (Large Language Model)** với **cơ sở dữ liệu phần cứng thực tế**  
nhằm đưa ra đề xuất laptop **chính xác – khách quan – phù hợp ngân sách và nhu cầu sử dụng**.

---

## 🚀 Tính năng nổi bật

### 🧠 Logic phần cứng thế hệ mới
- **Hỗ trợ RTX 5000 Series**  
  Tích hợp hệ thống đánh giá hiệu năng cho GPU Nvidia Blackwell mới nhất  
  *(RTX 5050 → RTX 5090)*, đồng thời duy trì thang điểm nhất quán với các thế hệ trước.

- **Adaptive Form-Factor Scoring**  
  Cơ chế chấm điểm linh hoạt theo hình thái thiết kế:
  - Laptop gaming chuyên dụng (trọng lượng ≥ 2.3kg) được **bù điểm hiệu năng** để phản ánh đúng khả năng tản nhiệt và công suất thực.
  - Laptop mỏng nhẹ và văn phòng được đánh giá cao hơn về **tính di động và trải nghiệm sử dụng dài hạn**.

- **Purpose-Aware Hardware Evaluation**  
  Phần cứng (CPU, GPU, RAM, màn hình) được đánh giá khác nhau tùy theo mục đích sử dụng  
  *(Gaming / Office / Creator / Thin & Light)*, giúp kết quả luôn phù hợp với nhu cầu thực tế.

- **Unified Scoring Engine**  
  Hệ thống chấm điểm độ phù hợp (Fit Score) được **đồng nhất logic giữa Backend và UI**.

---

### 🔍 Truy xuất dữ liệu & Xử lý thông minh
- **Vector Search (Semantic Search)**  
  Sử dụng **ChromaDB** để tìm kiếm laptop theo *ý nghĩa ngữ cảnh*,  
  không chỉ dựa trên khớp từ khóa.
- **Hybrid Filtering**  
  Kết hợp:
  - Lọc cứng theo **ngân sách**
  - Lọc theo **thương hiệu**
  → đảm bảo kết quả luôn nằm đúng tầm giá người dùng.
- **Smart Intent Detection**  
  Tự động nhận diện mục đích sử dụng:
  **Gaming – Văn phòng – Đồ họa – Mỏng nhẹ** từ câu hỏi ngôn ngữ tự nhiên.

---

## 🛠️ Danh mục công nghệ

| Thành phần | Công nghệ sử dụng |
| :--- | :--- |
| **Frontend UI** | Streamlit |
| **Vector Database** | ChromaDB |
| **LLM Models** | Groq (Llama 3.x) & Google Gemini |
| **Backend Logic** | Python |
| **Data Processing** | Pandas, NumPy, Pydantic |
| **Environment** | python-dotenv |
| **Data Source** | Google Sheets API & Local JSON |

---

## 📦 Hướng dẫn cài đặt & chạy dự án

1️. Khởi tạo môi trường
Yêu cầu **Python 3.9+**

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt

Tạo file .env tại thư mục gốc của dự án:

GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
ADMIN_PASSWORD=your_password

3. Xây dựng Vector Database

Chuyển đổi dữ liệu laptop từ JSON sang Vector để phục vụ RAG:
python scripts/create_vector_db.py

4. Khởi chạy ứng dụng

Mở giao diện Web bằng Streamlit:
streamlit run app.py

📂 Kiến trúc thư mục
CHATBOTLAPTOP/
│
├── app.py
├── run_all.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── chatbot_rag_core.py
│   └── __init__.py
│
├── scripts/
│   ├── data_standardization.py
│   └── create_vector_db.py
│
├── data/
│   ├── products_final.jsonl
│   ├── datalaptop.json
│   └── chroma_db/
│
└── .streamlit/
    └── secrets.toml

🛡️ Bảo mật & Ghi chú triển khai:
Các tệp chứa API Key, Service Account, Vector DB và log người dùng
đã được cấu hình trong .gitignore.

Khi cập nhật dữ liệu laptop mới, vui lòng:
python scripts/create_vector_db.py
để tái tạo chỉ mục tìm kiếm Vector.

Dự án được xây dựng theo hướng:
+Phân tách rõ Backend xử lý và Frontend hiển thị
+Logic chấm điểm minh bạch, có thể giải thích
Phù hợp cho:
+Báo cáo môn học
+Demo hệ thống AI ứng dụng thực tế
+Mở rộng thành sản phẩm hoàn chỉnh

📊 Nguồn dữ liệu tham khảo

Dữ liệu sản phẩm laptop trong dự án được **tham khảo và tổng hợp từ các nguồn công khai**, chủ yếu bao gồm:

- **Thế Giới Di Động**  
  https://www.thegioididong.com/laptop  

Các thông tin được sử dụng bao gồm:
- Tên sản phẩm
- Giá bán tham khảo tại thời điểm thu thập
- Thông số kỹ thuật cơ bản (CPU, GPU, RAM, SSD, màn hình, trọng lượng…)

🔒 **Lưu ý về dữ liệu**:
- Dữ liệu được sử dụng **chỉ phục vụ mục đích học tập, nghiên cứu và demo hệ thống**, không nhằm mục đích thương mại.
- Giá bán và tình trạng sản phẩm có thể thay đổi theo thời gian.
- Dự án **không liên kết, không đại diện và không có quan hệ thương mại** với Thế Giới Di Động.