import os
from decord import VideoReader, cpu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.video as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from torch.utils.data import Dataset, DataLoader
import time
from tqdm import tqdm

# 이진 분류를 위한 라벨 맵
label_map = {"Normal": 0, "Abnormal": 1}


def load_video_frames(video_path, num_frames=16):
    """
    영상 파일에서 일정한 간격으로 프레임을 추출합니다.
    """
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        indices = torch.linspace(0, total_frames - 1, num_frames).long()
        frames_nd = vr.get_batch(indices)
        frames = torch.from_numpy(frames_nd.asnumpy())
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
        return frames / 255.0
    except Exception as e:
        # print(f"영상 로드 중 오류 발생: {e}")
        return None


def extract_feature(frames, feature_extractor, device):
    """
    3D CNN 모델을 사용하여 영상 프레임에서 특징을 추출합니다.
    """
    if frames is None:
        return None

    frames = torch.nn.functional.interpolate(frames, size=(112, 112))
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)

    with torch.no_grad():
        # tqdm을 사용하여 프레임 처리 진행 상황 표시
        feat = feature_extractor(frames.to(device))
    return feat.view(-1).cpu()


def load_dataset(root_folder, feature_extractor, device, cache_folder="cache_features_add"):
    """
    캐싱 기능을 포함하여 데이터셋을 로드하고 특징을 추출합니다.
    """
    xs, ys = [], []
    os.makedirs(cache_folder, exist_ok=True)

    # Normal과 Abnormal 폴더만 처리
    for label_name, label_idx in label_map.items():
        folder = os.path.join(root_folder, label_name)
        if not os.path.isdir(folder):
            print(f"경로를 찾을 수 없음: {folder}. 이 폴더를 건너뜁니다.")
            continue

        video_files = [f for f in os.listdir(folder) if f.lower().endswith(('.avi', '.mp4', '.mov', '.mkv'))]
        print(f"\n{label_name}: {len(video_files)} videos")

        # tqdm을 사용하여 비디오 파일 처리 진행 상황 표시
        for vf in tqdm(video_files, desc=f"Processing {label_name} videos"):
            video_path = os.path.join(folder, vf)
            cache_path = os.path.join(cache_folder, f"{label_name}_{vf}.pt")

            if os.path.exists(cache_path):
                feature = torch.load(cache_path)
            else:
                frames = load_video_frames(video_path)
                if frames is not None:
                    feature = extract_feature(frames, feature_extractor, device)
                    torch.save(feature, cache_path)
                else:
                    continue

            xs.append(feature)
            ys.append(label_idx)

    return xs, ys


# Pre-trained 3D CNN 모델 준비
model_3d = models.r2plus1d_18(weights=models.R2Plus1D_18_Weights.KINETICS400_V1)
model_3d.eval()
feature_extractor = torch.nn.Sequential(*list(model_3d.children())[:-1])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.to(device)

# 데이터셋 로딩 (함수 호출 시 feature_extractor와 device 전달)
dataset_folder = "Dataset"
train_x, train_y = load_dataset(dataset_folder, feature_extractor, device)

# 데이터 분할
train_x, test_x, train_y, test_y = train_test_split(
    train_x, train_y, test_size=0.2, stratify=train_y, random_state=42
)

print(f"\nTrain samples: {len(train_x)}, Test samples: {len(test_x)}")


# Dataset 클래스
class VideoFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.stack(features, dim=0).to(torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 훈련/검증 데이터 분할
train_x_, val_x, train_y_, val_y = train_test_split(
    train_x, train_y, test_size=0.2, stratify=train_y, random_state=42
)


# MLP 분류기 (입력 레이어 크기 512로 변경)
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

        self.out = nn.Linear(32, 2)  # 이진 분류를 위해 출력 레이어 크기를 2로 유지

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


model_clf = ResidualMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_clf.parameters(), lr=1e-5)

# DataLoader 준비
train_dataset = VideoFeatureDataset(train_x_, train_y_)
val_dataset = VideoFeatureDataset(val_x, val_y)
test_dataset = VideoFeatureDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# EarlyStopping 설정
best_val_loss = float('inf')
patience = 10
patience_counter = 0
best_model_state = None

# 학습 루프
for epoch in range(1000):
    model_clf.train()
    train_loss = 0
    # tqdm을 사용하여 훈련 데이터셋 루프 진행 상황 표시
    with tqdm(train_loader, desc=f"Epoch {epoch + 1} Train", unit="batch") as t:
        for x, y in t:
            x, y = x.to(device), y.to(device)
            out = model_clf(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            t.set_postfix(loss=loss.item())

    train_loss /= len(train_loader.dataset)

    # Validation
    model_clf.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    # tqdm을 사용하여 검증 데이터셋 루프 진행 상황 표시
    with torch.no_grad():
        with tqdm(val_loader, desc=f"Epoch {epoch + 1} Val", unit="batch") as t:
            for x, y in t:
                x, y = x.to(device), y.to(device)
                out = model_clf(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                t.set_postfix(loss=loss.item())

    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)

    print(f"\nEpoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model_clf.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Best 모델 로드
if best_model_state is not None:
    model_clf.load_state_dict(best_model_state)

# Test 평가
model_clf.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    # tqdm을 사용하여 테스트 데이터셋 루프 진행 상황 표시
    with tqdm(test_loader, desc="Test Evaluation", unit="batch") as t:
        for x, y in t:
            x, y = x.to(device), y.to(device)
            out = model_clf(x)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

print("🧾 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
print("📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Abnormal"]))

# 모델 저장
torch.save(model_clf.state_dict(), 'camera_streamer/best_residual_mlp_model.pth')