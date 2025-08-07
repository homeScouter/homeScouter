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

label_map = {"Normal": 0, "Violence": 1, "Weaponized": 2}


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
        print(f"영상 로드 중 오류 발생: {e}")
        return None


def extract_feature(frames):
    """
    3D CNN 모델을 사용하여 영상 프레임에서 특징을 추출합니다.
    """
    if frames is None:
        return None

    frames = torch.nn.functional.interpolate(frames, size=(112, 112))
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
    device = next(feature_extractor.parameters()).device
    with torch.no_grad():
        feat = feature_extractor(frames.to(device))
    return feat.view(-1).cpu()


def load_dataset(root_folder, cache_folder="cache_features"):
    """
    캐싱 기능을 포함하여 데이터셋을 로드하고 특징을 추출합니다.
    """
    xs, ys = [], []
    os.makedirs(cache_folder, exist_ok=True)

    for label_name, label_idx in label_map.items():
        folder = os.path.join(root_folder, label_name)
        video_files = [f for f in os.listdir(folder) if f.endswith(".avi")]
        print(f"{label_name}: {len(video_files)} videos")

        for vf in video_files:
            video_path = os.path.join(folder, vf)
            cache_path = os.path.join(cache_folder, f"{label_name}_{vf}.pt")

            if os.path.exists(cache_path):
                feature = torch.load(cache_path)
                print(f"Loaded cache for {vf}")
            else:
                start = time.time()
                frames = load_video_frames(video_path)
                if frames is not None:
                    feature = extract_feature(frames)
                    torch.save(feature, cache_path)
                    end = time.time()
                    print(f"Processed {vf} in {end - start:.2f} sec and saved to cache")
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

# 데이터셋 로딩
train_folder = "SCVD_converted\Train"
test_folder = "SCVD_converted\Test"

train_x, train_y = load_dataset(train_folder)
test_x, test_y = load_dataset(test_folder)

print(f"Train samples: {len(train_x)}, Test samples: {len(test_x)}")


# Dataset 클래스
class VideoFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.stack(features, dim=0).to(torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def convert_to_binary(labels):
    return [0 if y == 0 else 1 for y in labels]


# 이진 분류를 위한 라벨 변환
train_y_bin = convert_to_binary(train_y)
test_y_bin = convert_to_binary(test_y)

# 훈련/검증 데이터 분할 (PCA 제거)
train_x_list = train_x  # PCA 없이 원본 리스트 사용
train_x_, val_x, train_y_, val_y = train_test_split(
    train_x_list, train_y_bin, test_size=0.2, stratify=train_y_bin, random_state=42
)


# `val_y`는 이미 `train_y_bin`에서 분리되었으므로 변환 불필요

# MLP 분류기 (입력 레이어 크기 512로 변경)
class ResidualMLP(nn.Module):
    def __init__(self, dropout_rate=0.4):
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


model_clf = ResidualMLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_clf.parameters(), lr=1e-6)

# DataLoader 준비 (PCA 제거된 데이터 사용)
train_dataset = VideoFeatureDataset(train_x_, train_y_)
val_dataset = VideoFeatureDataset(val_x, val_y)
test_dataset = VideoFeatureDataset(test_x, test_y_bin)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# EarlyStopping 설정
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# 학습 루프
for epoch in range(1000):
    model_clf.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model_clf(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model_clf.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model_clf(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)

    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

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
model_clf.load_state_dict(best_model_state)

# Test 평가
model_clf.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in test_loader:
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
