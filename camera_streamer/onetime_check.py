import os
import cv2
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from decord import VideoReader, cpu
import torchvision.models.video as models
from settings import RTSP_URL, CAPTURE_INTERVAL  # settings.py 파일이 있다고 가정


# -----------------------------------------------------------
# 1. 이전 코드에서 사용한 유틸리티 함수 및 모델 정의
# -----------------------------------------------------------

def load_video_frames(video_path, num_frames=16):
    """
    영상 파일에서 일정한 간격으로 프레임을 추출합니다.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        # 총 프레임 수에 맞춰 일정한 간격으로 16개 프레임 선택
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
        frames_nd = vr.get_batch(indices)
        frames = torch.from_numpy(frames_nd.asnumpy())
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W] -> [16, 3, H, W]
        return frames / 255.0
    except Exception as e:
        print(f"영상 파일 로드 중 오류 발생: {e}")
        return None


# 3D CNN 특징 추출기
model_3d = models.r2plus1d_18(weights=models.R2Plus1D_18_Weights.KINETICS400_V1)
model_3d.eval()
feature_extractor = torch.nn.Sequential(*list(model_3d.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)


def extract_feature(frames):
    """
    3D CNN 모델을 사용하여 영상 프레임에서 특징을 추출합니다.
    """
    if frames is None:
        return None

    # decord로 불러온 프레임 크기를 3D CNN에 맞게 조절
    frames = torch.nn.functional.interpolate(frames, size=(112, 112))
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
    with torch.no_grad():
        feat = feature_extractor(frames.to(device))
    return feat.view(-1).cpu().numpy().reshape(1, -1)  # [1, 512] 형태로 반환


# Residual MLP 분류기 모델
class ResidualMLP(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(512, 64)  # <-- 입력 차원을 512로 변경
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


# -----------------------------------------------------------
# 2. 모델 로드 (MLP만)
# -----------------------------------------------------------

# MLP 모델 로드
model_clf = ResidualMLP().to(device)
try:
    model_clf.load_state_dict(torch.load('best_residual_mlp_model.pth', map_location=device))
    model_clf.eval()
    print("✅ MLP 모델 가중치 로드 완료")
except FileNotFoundError:
    print("❌ best_residual_mlp_model.pth 파일을 찾을 수 없습니다. 모델을 먼저 학습시키고 저장하세요.")
    exit()


# -----------------------------------------------------------
# 3. RTSP 영상 캡처 및 예측 파이프라인
# -----------------------------------------------------------

def analyze_captured_video(video_path):
    """
    저장된 영상을 불러와 학습된 모델로 예측을 수행합니다.
    """
    print(f"🔍 영상 분석 시작: {video_path}")

    # 1. decord로 영상 프레임 로드
    frames = load_video_frames(video_path)
    if frames is None:
        return

    # 2. 3D CNN을 사용해 특징 추출 (512차원)
    feature_vector = extract_feature(frames)

    # 3. MLP 분류기로 예측 (PCA 단계 삭제)
    with torch.no_grad():
        x = torch.from_numpy(feature_vector).float().to(device)
        output = model_clf(x)
        probabilities = F.softmax(output, dim=1)
        pred_label_idx = torch.argmax(probabilities, dim=1).item()

    # 4. 결과 출력
    label_map_inv = {0: "정상 (Normal)", 1: "비정상 (Abnormal)"}
    predicted_label = label_map_inv[pred_label_idx]

    print(f"\n✨ 예측 결과: {predicted_label}")
    print(f"확률: {probabilities.cpu().numpy()[0]}")


def preview_captured_video(video_path):
    """
    저장된 영상을 화면에 띄워 임시로 재생합니다.
    'q' 키를 누르면 종료됩니다.
    """
    print(f"📺 '{video_path}' 영상 미리보기. 'q' 키를 눌러 종료하세요.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상 파일을 열 수 없습니다: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Captured Video Preview', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# RTSP 캡처 및 분석 메인 로직
rtsp_url = RTSP_URL
capture_duration = 4  # 4초
output_file = 'output_4sec.mp4'

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("❌ RTSP 연결 실패. URL을 확인해 주세요.")
else:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    start_time = time.time()
    print(f"▶️ {capture_duration}초 동안 영상 캡처 시작...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 실패")
            break

        out.write(frame)
        if time.time() - start_time > capture_duration:
            break

    cap.release()
    out.release()
    print(f"✅ {capture_duration}초짜리 영상 저장 완료: {output_file}")

    # 캡처된 영상 미리보기
    preview_captured_video(output_file)

    # 이거 영상 저장하는거는 비정상만 하고
    # 나중에 코드 구글 클라우드에 업로드하면, fireDB에 영상저장 할건데 그렇게할 코드 추가
    # 영상 저장 후 분석 함수 호출
    analyze_captured_video(output_file)
