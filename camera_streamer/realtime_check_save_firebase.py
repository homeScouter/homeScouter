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

# .env 파일 로드 (환경 변수 사용 시 필요)
load_dotenv()

# ---------------- Firebase Admin 초기화 ----------------
import firebase_admin
from firebase_admin import credentials, storage

if not firebase_admin._apps:
    cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

    if not cred_path:
        print("❌ 오류: 환경변수 'GOOGLE_APPLICATION_CREDENTIALS'가 설정되어 있지 않습니다.")
        print("Firebase Admin SDK 초기화에 실패했습니다. 프로그램을 종료합니다.")
        exit()  # 환경변수 없으면 강제 종료

    if not os.path.isfile(cred_path):
        print(f"❌ 오류: 서비스 계정 키 파일이 존재하지 않습니다: {cred_path}")
        print("Firebase Admin SDK 초기화에 실패했습니다. 프로그램을 종료합니다.")
        exit()  # 파일 없으면 강제 종료

    try:
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'home-scouter-50835.firebasestorage.app',
        })
        print("✅ Firebase Admin SDK 초기화 완료")
    except Exception as e:
        print(f"❌ Firebase Admin SDK 초기화 중 치명적인 오류 발생: {e}")
        print("프로그램을 종료합니다.")
        exit()  # 초기화 실패 시 강제 종료
else:
    print("Firebase Admin SDK는 이미 초기화되어 있습니다.")


# ------------- Firebase 업로드 함수 (에러 로깅 강화) -------------
def upload_to_firebase(local_path, cloud_path):
    """
    로컬 파일을 Firebase Storage에 업로드하고 다운로드 URL을 반환합니다.
    업로드 성공 시 로컬 파일을 삭제합니다.
    """
    try:
        bucket = storage.bucket()
        blob = bucket.blob(cloud_path)
        blob.upload_from_filename(local_path)  # 파일을 읽어 업로드
        print(f"✅ '{local_path}' → Firebase Storage '{cloud_path}' 업로드 완료.")

        # 업로드 성공 후 로컬 파일 삭제 시도.
        # WinError 32를 방지하기 위해 파일 핸들 클로즈가 선행되어야 함.
        os.remove(local_path)
        print(f"🗑️ '{local_path}' 임시 파일 삭제 완료.")

        # 서명된 URL 생성 (만료 시간 설정, 여기서는 1시간)
        url = blob.generate_signed_url(version="v4", expiration=3600)
        return url
    except Exception as e:
        print(f"❌ Firebase 업로드 실패: '{local_path}' → '{cloud_path}'")
        print(f"❌ 오류 상세: {e}")
        return None  # 업로드 실패 시 URL 반환하지 않음


# ------------- 모델 정의 및 로드 -------------
# 3D CNN 특징 추출기
model_3d = models.r2plus1d_18(weights=models.R2Plus1D_18_Weights.KINETICS400_V1)
model_3d.eval()
feature_extractor = torch.nn.Sequential(*list(model_3d.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)


def extract_feature(frames_tensor):
    """
    3D CNN 모델을 사용하여 텐서 형태의 프레임에서 특징을 추출합니다.
    """
    # decord로 불러온 프레임 크기를 3D CNN에 맞게 조절
    frames = torch.nn.functional.interpolate(frames_tensor, size=(112, 112))
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        feat = feature_extractor(frames.to(device))
    return feat.view(-1).cpu().numpy().reshape(1, -1)


# Residual MLP 분류기 모델
class ResidualMLP(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(512, 64)
        self.norm1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, 32)
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
    # 'best_residual_mlp_model.pth' 모델 파일은 이 스크립트와 같은 디렉토리에 있어야 합니다.
    model_clf.load_state_dict(torch.load('best_residual_mlp_model.pth', map_location=device))
    model_clf.eval()
    print("✅ MLP 모델 가중치 로드 완료")
except FileNotFoundError:
    print("❌ best_residual_mlp_model.pth 파일을 찾을 수 없습니다. 모델을 먼저 학습시키고 저장하세요.")
    print("프로그램을 종료합니다.")
    exit()  # 모델 파일 없으면 프로그램 종료


# -----------------------------------------------------------
# 2. 멀티스레딩을 사용한 RTSP 스트림 처리
# -----------------------------------------------------------

class RTSPFrameGrabber(threading.Thread):
    def __init__(self, rtsp_url, frame_queue):
        threading.Thread.__init__(self)
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.cap = None  # 초기에는 None으로 설정
        self.running = True
        self._connect_camera()

    def _connect_camera(self):
        """카메라 연결을 시도하는 내부 메서드"""
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.rtsp_url)
        if not self.cap.isOpened():
            print(f"⚠️ 카메라 연결 실패: {self.rtsp_url}")
        else:
            print(f"✅ 카메라 연결 성공: {self.rtsp_url}")

    def run(self):
        print("🎥 RTSP 프레임 캡처 스레드 시작...")
        retry_count = 0
        MAX_RETRIES = 10  # 최대 재연결 시도 횟수
        RETRY_DELAY = 5  # 재연결 시도 전 대기 시간 (초)

        while self.running:
            if not self.cap or not self.cap.isOpened():
                if retry_count >= MAX_RETRIES:
                    print("🚫 최대 재연결 시도 횟수 초과. RTSP 프레임 캡처 스레드를 종료합니다.")
                    self.running = False
                    break
                print(f"프레임 수신 불가 또는 카메라 연결 끊김. 재연결 시도 중... ({retry_count + 1}/{MAX_RETRIES})")
                self._connect_camera()  # 카메라 재연결 시도
                retry_count += 1
                time.sleep(RETRY_DELAY)
                continue

            try:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"프레임 수신 실패. 카메라 재연결 시도 중... ({retry_count + 1}/{MAX_RETRIES})")
                    self._connect_camera()  # 재연결 시도
                    retry_count += 1
                    time.sleep(RETRY_DELAY)
                    continue

                # 프레임 수신 성공 시 재연결 카운트 초기화
                retry_count = 0
                # 큐에 프레임 저장
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # 큐가 가득 찼을 경우 가장 오래된 프레임을 버리고 새 프레임을 추가
                    # print("⚠️ 프레임 큐가 가득 찼습니다. 가장 오래된 프레임을 버립니다.")
                    self.frame_queue.get_nowait()
                    self.frame_queue.put(frame)

            except cv2.error as e:
                print(f"❌ OpenCV 오류 발생: {e}. 카메라 재연결 시도 중...")
                self._connect_camera()  # OpenCV 오류 시 재연결
                retry_count += 1
                time.sleep(RETRY_DELAY)
            except Exception as e:
                print(f"❌ RTSP 캡처 중 예상치 못한 오류: {e}. 스레드 종료.")
                self.running = False
                break

    def stop(self):
        """스레드 중지 및 리소스 해제"""
        self.running = False
        if self.cap:
            self.cap.release()
        print("🛑 RTSP 프레임 캡처 스레드 종료.")


def process_stream_continuously(rtsp_url, interval_sec=CAPTURE_INTERVAL, num_frames_per_interval=16):
    """
    RTSP 스트림을 실시간으로 읽고, 지정된 간격마다 분석을 수행합니다.
    """
    # 큐 크기: 예를 들어, 30fps 스트림에서 4초 분량 (120 프레임)을 충분히 담을 수 있는 크기
    frame_queue = queue.Queue(maxsize=120)

    # 캡처 스레드 시작
    grabber = RTSPFrameGrabber(rtsp_url, frame_queue)
    grabber.start()

    last_analysis_time = time.time()
    label_map_inv = {0: "정상 (Normal)", 1: "비정상 (Abnormal)"}
    frame_buffer = []

    print(f"▶️ 실시간 분석 메인 스레드 시작 (분석 간격: {interval_sec}초)...")

    try:
        while True:
            # 캡처 스레드가 중단되면 메인 루프도 중단
            if not grabber.running and frame_queue.empty():
                print("메인 스레드: RTSP 캡처 스레드가 종료되어 분석을 중단합니다.")
                break

            # 큐에서 프레임 가져오기 (논블로킹)
            try:
                frame = frame_queue.get(block=False)
                frame_buffer.append(frame)
            except queue.Empty:
                pass  # 큐가 비어있으면 다음 루프로 넘어감 (프레임이 들어올 때까지 대기하지 않음)

            current_time = time.time()
            # 지정된 분석 간격이 되었거나, 버퍼에 충분한 프레임이 쌓였을 경우 분석 시작
            if current_time - last_analysis_time >= interval_sec:
                print(f"\n--- {interval_sec}초 영상 분석 시작 ---")

                if len(frame_buffer) > 0:
                    frames_to_analyze = []
                    # 버퍼의 프레임 수가 num_frames_per_interval보다 많을 경우 등간격으로 샘플링
                    step = max(1, len(frame_buffer) // num_frames_per_interval)
                    for i in range(num_frames_per_interval):
                        idx = min(i * step, len(frame_buffer) - 1)
                        frames_to_analyze.append(frame_buffer[idx])

                    frames_np = np.array(frames_to_analyze)
                    # OpenCV는 BGR, PyTorch 모델은 RGB를 예상할 수 있으므로 변환 필요
                    # R2plus1D_18 모델은 일반적으로 0-1 범위의 RGB 이미지를 예상합니다.
                    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).to(torch.float32) / 255.0
                    # BGR to RGB 변환 (만약 모델이 RGB를 예상한다면)
                    # frames_tensor = frames_tensor[:, [2, 1, 0], :, :]

                    feature_vector = extract_feature(frames_tensor)

                    with torch.no_grad():
                        x = torch.from_numpy(feature_vector).float().to(device)
                        output = model_clf(x)
                        probabilities = F.softmax(output, dim=1)
                        pred_label_idx = torch.argmax(probabilities, dim=1).item()

                    predicted_label = label_map_inv[pred_label_idx]
                    print(f"✨ 예측 결과: {predicted_label} | 확률: {probabilities.cpu().numpy()[0]}")

                    if pred_label_idx == 1:  # 'Abnormal' 감지
                        print("🚨 비정상 감지! Firebase Storage에 이미지/영상 저장합니다.")
                        timestamp = int(time.time())
                        image_filename = f'abnormal_event_{timestamp}.jpg'
                        video_filename = f'abnormal_video_{timestamp}.avi'

                        # Firebase Storage의 경로 설정 (폴더 구조)
                        firebase_image_path = f"abnormal_events/{image_filename}"
                        firebase_video_path = f"abnormal_videos/{video_filename}"

                        # ⭐️⭐️⭐️ 비정상 감지 처리 로직을 try-except로 감싸서 오류 발생 시에도 계속 실행되도록 함 ⭐️⭐️⭐️
                        # 또한, 파일 핸들을 명시적으로 닫아 WinError 32를 방지합니다.
                        img_tmp_file = None  # 초기화 (finally 블록에서 참조할 수 있도록)
                        vid_tmp_file = None  # 초기화
                        try:
                            # 1. 이미지 저장 및 업로드
                            representative_frame = frame_buffer[len(frame_buffer) // 2]

                            # 임시 파일로 저장 후 업로드 (업로드 후 자동 삭제)
                            img_tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                            cv2.imwrite(img_tmp_file.name, representative_frame)
                            img_tmp_file.close()  # ⭐️ 중요: 파일 핸들 명시적으로 닫기! (WinError 32 방지)

                            img_url = upload_to_firebase(img_tmp_file.name, firebase_image_path)

                            if img_url:
                                print(f"🌐 이미지 다운로드 URL: {img_url}")
                            else:
                                print("❌ 이미지 업로드 실패. URL 생성 안됨.")

                            # 2. 비디오 저장 및 업로드
                            # 비디오 파일 저장 (코덱 'XVID' 사용, FPS 20.0)
                            vid_tmp_file = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
                            fourcc = cv2.VideoWriter_fourcc(*'XVID')
                            # 비디오 해상도는 대표 프레임의 크기를 따름
                            out = cv2.VideoWriter(vid_tmp_file.name, fourcc, 20.0,
                                                  (representative_frame.shape[1], representative_frame.shape[0]))
                            for frame in frame_buffer:
                                out.write(frame)
                            out.release()  # 비디오 라이터 객체 해제
                            vid_tmp_file.close()  # ⭐️ 중요: 파일 핸들 명시적으로 닫기! (WinError 32 방지)

                            video_url = upload_to_firebase(vid_tmp_file.name, firebase_video_path)

                            if video_url:
                                print(f"🌐 비디오 다운로드 URL: {video_url}")
                            else:
                                print("❌ 비디오 업로드 실패. URL 생성 안됨.")

                        except Exception as e:
                            print(f"🚨 비정상 이벤트 처리 (저장/업로드) 중 오류 발생: {e}")
                            print("➡️ 이 오류는 무시하고 스트림 분석을 계속합니다.")
                        finally:
                            # 오류가 발생했거나 업로드 함수가 None을 반환했을 때도 임시 파일 삭제 시도
                            # 이미 close()를 했으므로 os.remove만 시도
                            if img_tmp_file and os.path.exists(img_tmp_file.name):
                                try:
                                    os.remove(img_tmp_file.name)
                                    print(f"🗑️ Finally block: '{img_tmp_file.name}' 삭제 완료.")
                                except OSError as e:
                                    print(f"⚠️ Finally block: 이미지 임시 파일 삭제 실패: {e}")

                            if vid_tmp_file and os.path.exists(vid_tmp_file.name):
                                try:
                                    os.remove(vid_tmp_file.name)
                                    print(f"🗑️ Finally block: '{vid_tmp_file.name}' 삭제 완료.")
                                except OSError as e:
                                    print(f"⚠️ Finally block: 비디오 임시 파일 삭제 실패: {e}")
                    else:  # 정상 상황
                        print("✅ 정상 상황입니다.")

                else:
                    print("⚠️ 프레임 버퍼가 비어있습니다. 다음 간격까지 대기합니다.")

                frame_buffer = []  # 이번 분석에 사용된 프레임 버퍼 비우기
                last_analysis_time = current_time  # 다음 분석 시작 시간을 현재 시간으로 설정

            # CPU 점유율을 낮추기 위한 sleep (너무 짧으면 CPU 점유율 높음, 너무 길면 프레임 놓칠 수 있음)
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n👋 사용자 요청으로 스트림을 종료합니다.")
    except Exception as e:
        print(f"❌ 메인 스레드에서 예상치 못한 오류 발생: {e}")
    finally:
        # 프로그램 종료 시 리소스 해제
        grabber.stop()
        grabber.join()  # 캡처 스레드가 완전히 종료될 때까지 대기
        cv2.destroyAllWindows()
        print("✅ 모든 리소스를 해제했습니다.")


if __name__ == "__main__":
    process_stream_continuously(RTSP_URL)
