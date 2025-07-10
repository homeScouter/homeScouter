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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Dataset 클래스
class VideoFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# SVM 분류기
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# torch tensor → numpy
X_train = torch.stack(train_x).numpy()
y_train = np.array(train_y)
X_test = torch.stack(test_x).numpy()
y_test = np.array(test_y)

# 얘는 삼진

#clf = SVC(kernel='rbf', C=1.0, class_weight='balanced')  # class imbalance 대응
#clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
#
# print("📊 Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("📄 Classification Report:")
# print(classification_report(y_test, y_pred, target_names=["Normal", "Violence", "Weaponed"]))

"""
# 얘는 이진, SVM 85% accuracy 나옴

# 1) 라벨 이진 변환 함수
def convert_to_binary_labels(labels):
    return np.array([0 if y == 0 else 1 for y in labels])  # Normal=0, 나머지=1

# 2) 변환 적용
y_train_bin = convert_to_binary_labels(train_y)
y_test_bin = convert_to_binary_labels(test_y)

clf = SVC(kernel='rbf', C=1.0, class_weight='balanced')  # class imbalance 대응
clf.fit(X_train, y_train_bin)

y_pred_bin = clf.predict(X_test)

print("📊 Confusion Matrix:")
print(confusion_matrix(y_test_bin, y_pred_bin))
print("📄 Classification Report:")
print(classification_report(y_test_bin, y_pred_bin, target_names=["Normal", "Abnormal"]))

"""

def convert_to_binary(labels):
    return [0 if y == 0 else 1 for y in labels]

train_y_bin = convert_to_binary(train_y)
test_y_bin = convert_to_binary(test_y)

train_x_, val_x, train_y_, val_y = train_test_split(
    train_x, train_y_bin, test_size=0.2, stratify=train_y_bin, random_state=42
)

val_y = convert_to_binary(val_y)

# MLP 분류기
model_clf = nn.Sequential(
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.LayerNorm(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.LayerNorm(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 2)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_clf.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_clf.parameters(), lr=1e-6)

train_dataset = VideoFeatureDataset(train_x_, train_y_)
val_dataset = VideoFeatureDataset(val_x, val_y)
test_dataset = VideoFeatureDataset(test_x, test_y)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

# 👉 EarlyStopping 설정
best_val_loss = float('inf')
patience = 5
patience_counter = 0

# 🔁 학습 루프
for epoch in range(200):
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

    # 🔎 Validation
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

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ✅ Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model_clf.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# 📌 Best 모델 로드
model_clf.load_state_dict(best_model_state)

# ✅ Test 평가
model_clf.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x, y in DataLoader(test_dataset, batch_size=4):
        x, y = x.to(device), y.to(device)
        out = model_clf(x)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

# 이진 라벨로 변환
all_labels_bin = [0 if y == 0 else 1 for y in all_labels]
all_preds_bin = [0 if y == 0 else 1 for y in all_preds]

print("🧾 Confusion Matrix:")
print(confusion_matrix(all_labels_bin, all_preds_bin))
print("📊 Classification Report:")
print(classification_report(all_labels_bin, all_preds_bin, target_names=["Normal", "Abnormal"]))
