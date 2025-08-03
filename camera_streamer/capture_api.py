# capture_api.py
import cv2
import numpy as np
from .cache import memory_cache
from .settings import RTSP_URL

def capture_single_frame(rtsp_url=RTSP_URL):
    """
    RTSP 스트림에서 한 장의 프레임을 캡처하여
    JPEG 바이너리 형태로 메모리 캐시에 저장하고 반환합니다.
    """
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("RTSP 연결 실패")
        return None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("프레임 수신 실패")
        return None

    # JPEG 인코딩
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        print("JPEG 인코딩 실패")
        return None

    jpeg_bytes = jpeg.tobytes()

    # 메모리 캐시에 10초간 저장
    memory_cache.set('latest_frame', jpeg_bytes, timeout=10)
    print("한 장의 프레임을 캡처하여 캐시에 저장했습니다.")

    return jpeg_bytes
