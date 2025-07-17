# capture.py
import cv2
import numpy as np
from time import sleep
from cache import memory_cache
from settings import RTSP_URL, CAPTURE_INTERVAL

def capture_video(rtsp_url, interval=CAPTURE_INTERVAL):
    """카메라 영상을 일정 간격으로 캡처하여 메모리에 저장하고 실시간으로 표시."""
    # RTSP 스트림 연결
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print("RTSP 연결 실패")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임 수신 실패")
            break

        # 지정된 간격으로 프레임 처리
        if frame_count % (interval * 30) == 0:  # Assuming 30 fps
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                memory_cache.set('latest_frame', jpeg.tobytes(), timeout=10)  # 10초 동안 저장
                print("Captured a new frame.")

        # 실시간 카메라 영상 표시
        cv2.imshow("Tapo C210 Stream", frame)

        # 최근 캡처된 이미지를 표시 (메모리 캐시에서 가져오기)
        latest_frame = memory_cache.get('latest_frame')
        if latest_frame:
            # 이미지를 디코딩하여 표시
            frame_image = np.frombuffer(latest_frame, dtype=np.uint8)
            frame_image = cv2.imdecode(frame_image, cv2.IMREAD_COLOR)
            cv2.imshow("Captured Frame", frame_image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        sleep(1 / 30)  # 30fps로 설정 (1초에 30 프레임)

    # 비디오 캡처 종료
    cap.release()
    cv2.destroyAllWindows()

