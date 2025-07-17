# main.py

from capture import capture_video
from settings import RTSP_URL

def main():
    # 카메라에서 영상을 4초 간격으로 캡처 시작
    print("Starting camera stream...")
    capture_video(RTSP_URL, interval=4)

if __name__ == "__main__":
    main()