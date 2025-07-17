import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# RTSP 카메라 URL (환경변수로부터 읽어옴)
RTSP_URL = os.getenv('RTSP_URL', 'rtsp://your_camera_ip/stream')  # 기본값 설정

# 카메라 캡처 간격 (환경변수에서 읽어오기, 기본값은 4초)
CAPTURE_INTERVAL = int(os.getenv('CAPTURE_INTERVAL', 4))
