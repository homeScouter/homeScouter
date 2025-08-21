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
from decord import VideoReader, cpu
from settings import RTSP_URL, CAPTURE_INTERVAL  # settings.py 파일이 있다고 가정

# -----------------------------------------------------------
# 1. 모델 정의 및 로드
# -----------------------------------------------------------

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


# TCN(Temporal Convolutional Network) 분류기 모델
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation_rate // 2,
                               dilation=dilation_rate)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation_rate // 2,
                               dilation=dilation_rate)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return F.relu(out + residual)


class TCNClassifier(nn.Module):
    def __init__(self, input_size=512, num_channels=[64, 128], kernel_size=3):
        super(TCNClassifier, self).__init__()
        self.blocks = nn.ModuleList([
            TCNBlock(input_size if i == 0 else num_channels[i - 1],
                     num_channels[i],
                     kernel_size,
                     2 ** i) for i in range(len(num_channels))
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], 2)

    def forward(self, x):
        # [batch_size, 512] -> [batch_size, 1, 512]
        x = x.unsqueeze(1)

        for block in self.blocks:
            x = block(x)

        x = self.pool(x).squeeze(-1)
        x = self.fc(x)

        return x


# TCN 모델 로드
model_clf = TCNClassifier().to(device)
try:
    model_clf.load_state_dict(torch.load('best_tcn_model.pth', map_location=device))
    model_clf.eval()
    print("✅ TCN 모델 가중치 로드 완료")
except FileNotFoundError:
    print("❌ best_tcn_model.pth 파일을 찾을 수 없습니다. 모델을 먼저 학습시키고 저장하세요.")
    exit()


# -----------------------------------------------------------
# 2. 멀티스레딩을 사용한 RTSP 스트림 처리
# -----------------------------------------------------------

class RTSPFrameGrabber(threading.Thread):
    def __init__(self, rtsp_url, frame_queue):
        threading.Thread.__init__(self)
        self.rtsp_url = rtsp_url
        self.frame_queue = frame_queue
        self.cap = cv2.VideoCapture(rtsp_url)
        self.running = True

    def run(self):
        print("🎥 RTSP 프레임 캡처 스레드 시작...")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("프레임 수신 실패 또는 스트림 종료. 재연결 시도...")
                self.cap.release()
                self.cap = cv2.VideoCapture(self.rtsp_url)
                time.sleep(1)
                continue

            # 큐에 프레임 저장
            self.frame_queue.put(frame)

    def stop(self):
        self.running = False
        self.cap.release()
        print("🛑 RTSP 프레임 캡처 스레드 종료.")


def process_stream_continuously(rtsp_url, interval_sec=4, num_frames_per_interval=16):
    """
    RTSP 스트림을 실시간으로 읽고, 지정된 간격마다 분석을 수행합니다.
    """
    frame_queue = queue.Queue(maxsize=120)  # 최대 120 프레임 (4초) 버퍼

    # 캡처 스레드 시작
    grabber = RTSPFrameGrabber(rtsp_url, frame_queue)
    grabber.start()

    last_analysis_time = time.time()
    label_map_inv = {0: "정상 (Normal)", 1: "비정상 (Abnormal)"}
    frame_buffer = []

    # 비정상 이벤트 이미지와 영상을 저장할 폴더 생성
    abnormal_events_folder = 'abnormal_events'
    abnormal_videos_folder = 'abnormal_videos'
    os.makedirs(abnormal_events_folder, exist_ok=True)
    os.makedirs(abnormal_videos_folder, exist_ok=True)
    print(f"📂 비정상 이벤트 이미지는 '{abnormal_events_folder}', 영상은 '{abnormal_videos_folder}' 폴더에 저장됩니다.")

    print("▶️ 실시간 분석 메인 스레드 시작...")

    try:
        while True:
            # 큐에서 프레임 가져오기 (논블로킹)
            try:
                frame = frame_queue.get(block=False)
                frame_buffer.append(frame)
            except queue.Empty:
                pass  # 큐가 비어있으면 다음 루프로 넘어감

            current_time = time.time()
            if current_time - last_analysis_time >= interval_sec:
                print(f"\n--- {interval_sec}초 영상 분석 시작 ---")

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

                    print(f"✨ 예측 결과: {predicted_label} | 확률: {probabilities.cpu().numpy()[0]}")

                    if pred_label_idx == 1:
                        print("🚨 'Abnormal' 감지! 대표 이미지와 영상을 저장합니다.")

                        # 파일 경로를 폴더와 파일명으로 조합
                        timestamp = int(time.time())
                        image_filename = f'abnormal_event_{timestamp}.jpg'
                        video_filename = f'abnormal_video_{timestamp}.avi'

                        image_filepath = os.path.join(abnormal_events_folder, image_filename)
                        video_filepath = os.path.join(abnormal_videos_folder, video_filename)

                        # 이미지 저장
                        representative_frame = frame_buffer[len(frame_buffer) // 2]
                        cv2.imwrite(image_filepath, representative_frame)
                        print(f"💾 '{image_filepath}' 파일로 저장했습니다.")

                        # 비디오 저장
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        out = cv2.VideoWriter(video_filepath, fourcc, 20.0,
                                              (frame_buffer[0].shape[1], frame_buffer[0].shape[0]))
                        for frame in frame_buffer:
                            out.write(frame)
                        out.release()
                        print(f"🎥 '{video_filepath}' 파일로 저장했습니다.")
                else:
                    print("⚠️ 프레임 버퍼가 비어있습니다. 다음 간격까지 대기합니다.")

                frame_buffer = []
                last_analysis_time = current_time

            # CPU 점유율을 낮추기 위한 sleep
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n👋 사용자 요청으로 스트림을 종료합니다.")
    finally:
        grabber.stop()
        grabber.join()
        cv2.destroyAllWindows()
        print("✅ 모든 리소스를 해제했습니다.")


if __name__ == "__main__":
    process_stream_continuously(RTSP_URL)
