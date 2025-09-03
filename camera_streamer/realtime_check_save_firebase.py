import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as models
import numpy as np
import threading
import queue
import tempfile
from dotenv import load_dotenv  # 환경 변수 로드를 위해 추가

from settings import RTSP_URL, CAPTURE_INTERVAL

# 로깅 초기화
import logging
logger = logging.getLogger('camera_streamer')  # 로거 이름 설정
logger.setLevel(logging.INFO)  # 로깅 레벨 설정 (INFO 이상의 로그만 출력)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# .env 파일 로드 (환경 변수 사용 시 필요)
load_dotenv()

# ---------------- Firebase Admin 초기화 ----------------
import firebase_admin
from firebase_admin import credentials, storage, firestore   ### NEW: Firestore 추가

if not firebase_admin._apps:
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    if not cred_path:
        logger.error("오류: 환경변수 'GOOGLE_APPLICATION_CREDENTIALS'가 설정되어 있지 않습니다.")
        exit()

    if not os.path.isfile(cred_path):
        logger.error(f"오류: 서비스 계정 키 파일이 존재하지 않습니다: {cred_path}")
        exit()

    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'home-scouter-50835.firebasestorage.app',
        })
        logger.info("Firebase Admin SDK 초기화 완료")
    except Exception as e:
        logger.error(f"Firebase Admin SDK 초기화 중 오류 발생: {e}")
        exit()
else:
    logger.info("Firebase Admin SDK는 이미 초기화되어 있습니다.")

# Firestore 클라이언트 초기화
db = firestore.client()

# ------------- Firebase 업로드 함수 (에러 로깅 강화) -------------
def upload_to_firebase(local_path, cloud_path):
    """
    로컬 파일을 Firebase Storage에 업로드하고 다운로드 URL을 반환합니다.
    업로드 성공 시 로컬 파일을 삭제합니다.
    """
    try:
        bucket = storage.bucket()
        blob = bucket.blob(cloud_path)
        blob.upload_from_filename(local_path)
        logger.info(f"'{local_path}' → Firebase Storage '{cloud_path}' 업로드 완료.")

        # 업로드 성공 후 로컬 파일 삭제
        os.remove(local_path)
        logger.info(f"'{local_path}' 임시 파일 삭제 완료.")

        # 서명된 URL 생성 (만료 시간 1시간)
        url = blob.generate_signed_url(version="v4", expiration=3600)
        return url
    except Exception as e:
        logger.error(f"Firebase 업로드 실패: '{local_path}' → '{cloud_path}'")
        logger.error(f"오류 상세: {e}")
        return None


# ------------- Firestore 저장 함수 -------------
def save_event_to_firestore(event_type, image_url, video_url, probability, timestamp):
    """
    Firestore에 이벤트 메타데이터 저장
    """
    try:
        doc_ref = db.collection("abnormal_events").document(str(timestamp))
        doc_ref.set({
            "event_type": event_type,
            "image_url": image_url,
            "video_url": video_url,
            "probability": probability.tolist() if probability is not None else None,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Firestore에 이벤트 저장 완료 (ID: {timestamp})")
    except Exception as e:
        logger.error(f"Firestore 저장 실패: {e}")


# ------------- 모델 정의 및 로드 -------------
# 3D CNN 특징 추출기
model_3d = models.r2plus1d_18(weights=models.R2Plus1D_18_Weights.KINETICS400_V1)
model_3d.eval()
feature_extractor = torch.nn.Sequential(*list(model_3d.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)

def extract_feature(frames_tensor):
    """3D CNN 모델을 사용하여 특징 추출"""
    frames = torch.nn.functional.interpolate(frames_tensor, size=(112, 112))
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        feat = feature_extractor(frames.to(device))
    return feat.view(-1).cpu().numpy().reshape(1, -1)


# Residual MLP 분류기 모델
class ResidualMLP(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(512, 128)
        self.norm1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(128, 128)
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, 32)
        self.norm3 = nn.LayerNorm(32)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.out = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = self.dropout1(x)
        residual = x
        x = F.relu(self.norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = x + residual
        x = F.relu(self.norm3(self.fc3(x)))
        x = self.dropout3(x)
        return self.out(x)


# MLP 모델 로드
model_clf = ResidualMLP().to(device)
try:
    model_clf.load_state_dict(torch.load('best_residual_mlp_model_new.pth', map_location=device))
    model_clf.eval()
    logger.info("MLP 모델 가중치 로드 완료")  # 특수문자 제거
except FileNotFoundError:
    logger.error("best_residual_mlp_model.pth 파일을 찾을 수 없습니다.")
    exit()


# -----------------------------------------------------------
# 2. 멀티스레딩을 사용한 RTSP 스트림 처리
# -----------------------------------------------------------

class RTSPFrameGrabber(threading.Thread):
    def __init__(self, rtsp_url, frame_queue):
        threading.Thread.__init__(self)
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.cap = None
        self.running = True
        self._connect_camera()

    def _connect_camera(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            logger.error(f"카메라 연결 실패: {self.rtsp_url}")
        else:
            logger.info(f"카메라 연결 성공: {self.rtsp_url}")

    def run(self):
        logger.info("RTSP 프레임 캡처 스레드 시작...")
        retry_count = 0
        MAX_RETRIES = 10
        RETRY_DELAY = 5

        while self.running:
            if not self.cap or not self.cap.isOpened():
                if retry_count >= MAX_RETRIES:
                    logger.error("최대 재연결 시도 초과.")
                    self.running = False
                    break
                logger.info(f"프레임 수신 불가, 재연결 시도... ({retry_count + 1}/{MAX_RETRIES})")
                self._connect_camera()
                retry_count += 1
                time.sleep(RETRY_DELAY)
                continue

            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info(f"프레임 수신 실패, 재연결 시도... ({retry_count + 1}/{MAX_RETRIES})")
                    self._connect_camera()
                    retry_count += 1
                    time.sleep(RETRY_DELAY)
                    continue

                retry_count = 0
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)

            except cv2.error as e:
                logger.error(f"OpenCV 오류: {e}")
                self._connect_camera()
                retry_count += 1
                time.sleep(RETRY_DELAY)
            except Exception as e:
                logger.error(f"RTSP 캡처 중 오류: {e}")
                self.running = False
                break

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("RTSP 프레임 캡처 스레드 종료.")


def process_stream_continuously(rtsp_url, interval_sec=CAPTURE_INTERVAL, num_frames_per_interval=16):
    """RTSP 스트림을 읽고, 지정된 간격마다 분석 (실시간 화면 출력 없음)"""
    frame_queue = queue.Queue(maxsize=120)
    grabber = RTSPFrameGrabber(rtsp_url, frame_queue)
    grabber.start()

    last_analysis_time = time.time()
    label_map_inv = {0: "정상 (Normal)", 1: "비정상 (Abnormal)"}
    frame_buffer = []

    logger.info(f"실시간 분석 시작 (간격: {interval_sec}초)...")

    try:
        while True:
            if not grabber.running and frame_queue.empty():
                logger.info("메인 스레드: 캡처 스레드 종료됨.")
                break

            try:
                frame = frame_queue.get(block=False)
                frame_buffer.append(frame)
            except queue.Empty:
                pass

            current_time = time.time()
            if current_time - last_analysis_time >= interval_sec:
                logger.info(f"\n--- {interval_sec}초 영상 분석 ---")

                if len(frame_buffer) > 0:
                    frames_to_analyze = []
                    step = max(1, len(frame_buffer) // num_frames_per_interval)
                    for i in range(num_frames_per_interval):
                        idx = min(i * step, len(frame_buffer) - 1)
                        frames_to_analyze.append(frame_buffer[idx])

                    frames_np = np.array(frames_to_analyze)
                    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(torch.float32) / 255.0

                    feature_vector = extract_feature(frames_tensor)

                    with torch.no_grad():
                        x = torch.from_numpy(feature_vector).float().to(device)
                        output = model_clf(x)
                        probabilities = F.softmax(output, dim=1)
                        pred_label_idx = torch.argmax(probabilities, dim=1).item()

                    predicted_label = label_map_inv[pred_label_idx]
                    logger.info(f"예측 결과: {predicted_label} | 확률: {probabilities.cpu().numpy()[0]}")

                    if pred_label_idx == 1:  # Abnormal
                        logger.info("비정상 감지! Firebase 업로드 + Firestore 저장")
                        timestamp = int(time.time())
                        image_filename = f'abnormal_event_{timestamp}.jpg'
                        video_filename = f'abnormal_video_{timestamp}.avi'

                        firebase_image_path = f"abnormal_events/{image_filename}"
                        firebase_video_path = f"abnormal_videos/{video_filename}"

                        img_tmp_file = None
                        vid_tmp_file = None
                        try:
                            representative_frame = frame_buffer[len(frame_buffer) // 2]
                            img_tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                            cv2.imwrite(img_tmp_file.name, representative_frame)
                            img_tmp_file.close()
                            img_url = upload_to_firebase(img_tmp_file.name, firebase_image_path)

                            vid_tmp_file = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            out = cv2.VideoWriter(vid_tmp_file.name, fourcc, 20.0,
                                                  (representative_frame.shape[1], representative_frame.shape[0]))
                            for frame in frame_buffer:
                                out.write(frame)
                            out.release()
                            vid_tmp_file.close()
                            video_url = upload_to_firebase(vid_tmp_file.name, firebase_video_path)

                            # Firestore 저장
                            save_event_to_firestore(
                                event_type="abnormal",
                                image_url=img_url,
                                video_url=video_url,
                                probability=probabilities.cpu().numpy()[0],
                                timestamp=timestamp
                            )

                        except Exception as e:
                            logger.error(f"비정상 이벤트 저장 중 오류: {e}")
                        finally:
                            if img_tmp_file and os.path.exists(img_tmp_file.name):
                                try: os.remove(img_tmp_file.name)
                                except: pass
                            if vid_tmp_file and os.path.exists(vid_tmp_file.name):
                                try: os.remove(vid_tmp_file.name)
                                except: pass
                    else:
                        logger.info("정상 상황입니다.")

                else:
                    logger.info("프레임 버퍼 비어있음.")

                frame_buffer = []
                last_analysis_time = current_time

            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("\n사용자 요청 종료")
    except Exception as e:
        logger.error(f"메인 스레드 오류: {e}")
    finally:
        grabber.stop()
        grabber.join()
        cv2.destroyAllWindows()
        logger.info("리소스 해제 완료")

if __name__ == "__main__":
    process_stream_continuously(RTSP_URL)
