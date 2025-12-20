"""
Script Quản Lý & Vận Hành Dự Án.

Quy trình tự động:
1. Chuẩn hóa dữ liệu (ETL) -> scripts/data_standardization.py
2. Tạo Vector Database       -> scripts/create_vector_db.py (Tên file cũ của bạn)
3. Khởi chạy Ứng dụng        -> app.py

Cách dùng:
    python run_all.py
"""

import subprocess
import sys
import time
from pathlib import Path
from typing import List

# =============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (CONFIGURATION)
# =============================================================================
# Thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT_DIR / "scripts"

# Định nghĩa đường dẫn các file script
# Lưu ý: Đã giữ nguyên tên file "create_vector_db.py" theo ý bạn
SCRIPT_STANDARDIZE = SCRIPTS_DIR / "data_standardization.py"
SCRIPT_VECTOR_DB = SCRIPTS_DIR / "create_vector_db.py" 
SCRIPT_APP = ROOT_DIR / "app.py"

# Trình thông dịch Python hiện tại (Đảm bảo dùng đúng môi trường ảo đang kích hoạt)
PYTHON_EXEC = sys.executable

# =============================================================================
# 2. GIAO DIỆN TERMINAL (MÀU SẮC & LOG)
# =============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log_header(msg: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}\n{msg}\n{'='*60}{Colors.ENDC}")

def log_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.ENDC}")

def log_success(msg: str, duration: float = 0):
    time_str = f"(Mất {duration:.2f}s)" if duration > 0 else ""
    print(f"{Colors.GREEN}✅ {msg} {time_str}{Colors.ENDC}")

def log_error(msg: str):
    print(f"{Colors.FAIL}❌ {msg}{Colors.ENDC}")

# =============================================================================
# 3. BỘ MÁY THỰC THI (EXECUTION ENGINE)
# =============================================================================
def check_file_exists(path: Path) -> bool:
    if not path.exists():
        log_error(f"Thiếu file quan trọng: {path}")
        return False
    return True

def run_command(command: List[str], step_name: str, block: bool = True) -> bool:
    """
    Hàm thực thi lệnh shell.
    block=True: Chờ chạy xong mới đi tiếp (Dùng cho ETL/DB).
    block=False: Chạy nền (Dùng cho Streamlit App).
    """
    start_time = time.time()
    
    log_info(f"Đang thực hiện: {step_name}...")
    
    try:
        if not block:
            # Chạy nền (cho App)
            subprocess.Popen(command, cwd=ROOT_DIR)
            return True

        # Chạy chờ kết quả (cho Script xử lý data)
        subprocess.run(command, check=True, cwd=ROOT_DIR)
        
        elapsed = time.time() - start_time
        log_success(f"Hoàn thành bước: {step_name}", elapsed)
        return True

    except subprocess.CalledProcessError as e:
        log_error(f"{step_name} thất bại với mã lỗi {e.returncode}.")
        return False
    except KeyboardInterrupt:
        log_error("Người dùng đã hủy quy trình.")
        return False
    except Exception as e:
        log_error(f"Lỗi khi chạy {step_name}: {e}")
        return False

# =============================================================================
# 4. QUY TRÌNH CHÍNH (MAIN PIPELINE)
# =============================================================================
def main():
    log_header("🚀 KHỞI ĐỘNG HỆ THỐNG AI LAPTOP CHATBOT")
    log_info(f"Thư mục gốc: {ROOT_DIR}")
    
    # 0. Kiểm tra file
    required_files = [SCRIPT_STANDARDIZE, SCRIPT_VECTOR_DB, SCRIPT_APP]
    if not all(map(check_file_exists, required_files)):
        log_error("Vui lòng kiểm tra lại cấu trúc thư mục 'scripts/' và file app.py.")
        sys.exit(1)

    # 1. Bước 1: Chuẩn hóa dữ liệu
    print("-" * 40)
    if not run_command([PYTHON_EXEC, str(SCRIPT_STANDARDIZE)], "Chuẩn hóa dữ liệu (Data Standardization)"):
        sys.exit(1)

    # 2. Bước 2: Tạo Database Vector
    print("-" * 40)
    if not run_command([PYTHON_EXEC, str(SCRIPT_VECTOR_DB)], "Tạo Vector Database (Embedding)"):
        sys.exit(1)

    # 3. Bước 3: Khởi chạy Giao diện
    print("-" * 40)
    log_header("🌐 ĐANG KHỞI CHẠY GIAO DIỆN STREAMLIT")
    
    streamlit_cmd = [
        PYTHON_EXEC, "-m", "streamlit", "run", 
        str(SCRIPT_APP),
        "--server.port=8501",
        "--theme.base=dark"
    ]
    
    if run_command(streamlit_cmd, "Streamlit App", block=False):
        log_success("Ứng dụng đang chạy! Hãy kiểm tra trình duyệt của bạn.")
        log_info("Nhấn Ctrl+C để dừng server.")
        
        # Giữ script sống để theo dõi
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            log_info("Đang tắt hệ thống...")

if __name__ == "__main__":
    main()