import os
from decord import VideoReader, cpu
import torch

label_map = {"Normal":0, "Violence":1, "Weaponized":2}

def load_video_frames(video_path, num_frames=16):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)
    indices = torch.linspace(0, total_frames - 1, num_frames).long()
    frames_nd = vr.get_batch(indices)
    frames = torch.from_numpy(frames_nd.asnumpy())
    frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]
    return frames / 255.0

def extract_feature(frames):
    frames = torch.nn.functional.interpolate(frames, size=(112,112))
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
    device = next(feature_extractor.parameters()).device
    with torch.no_grad():
        feat = feature_extractor(frames.to(device))
    return feat.view(-1).cpu()

import time

def load_dataset(root_folder, cache_folder="cache_features"):
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
                # 캐시가 존재하면 불러오기
                feature = torch.load(cache_path)
                print(f"Loaded cache for {vf}")
            else:
                # 아니면 추출 후 저장
                start = time.time()
                frames = load_video_frames(video_path)
                feature = extract_feature(frames)
                torch.save(feature, cache_path)
                end = time.time()
                print(f"Processed {vf} in {end - start:.2f} sec and saved to cache")

            xs.append(feature)
            ys.append(label_idx)

    return xs, ys

# pretrained 3D CNN 모델 준비
import torchvision.models.video as models
model_3d = models.r2plus1d_18(weights=models.R2Plus1D_18_Weights.KINETICS400_V1)
model_3d.eval()
feature_extractor = torch.nn.Sequential(*list(model_3d.children())[:-1])
feature_extractor = feature_extractor.cuda() if torch.cuda.is_available() else feature_extractor.cpu()

# 데이터셋 로딩
train_folder = "SCVD_converted\Train"
test_folder = "SCVD_converted\Test"

train_x, train_y = load_dataset(train_folder)
test_x, test_y = load_dataset(test_folder)

print(f"Train samples: {len(train_x)}, Test samples: {len(test_x)}")

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Dataset 정의
class VideoFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features  # list of torch.Tensor
        self.labels = labels      # list of int

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 모델 정의
model_clf = nn.Sequential(
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Linear(128, 3)  # 3 classes
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_clf.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_clf.parameters(), lr=1e-3)

# 데이터셋, DataLoader 준비
train_dataset = VideoFeatureDataset(train_x, train_y)
test_dataset = VideoFeatureDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 학습 루프
num_epochs = 50
for epoch in range(num_epochs):
    model_clf.train()
    total_loss = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        out = model_clf(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# 평가 (test)
model_clf.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        out = model_clf(x)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# Confusion matrix 출력
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

# 자세한 분류 리포트 출력
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Normal", "Violence", "Weaponed"]))
