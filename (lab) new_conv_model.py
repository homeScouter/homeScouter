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
        if isinstance(features, list):
            self.features = torch.stack(features).float()
        else:
            self.features = torch.tensor(features, dtype=torch.float32)

        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# SVM 분류기
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 🔹 1. Tensor → Numpy
X_train = torch.stack(train_x).numpy()
X_test = torch.stack(test_x).numpy()

# 🔹 2. 정규화 (PCA 전 필수)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 라벨 이진화
def convert_to_binary(labels):
    return [0 if y == 0 else 1 for y in labels]

y_train = np.array(convert_to_binary(train_y))
y_test = np.array(convert_to_binary(test_y))

# 🔹 3. 학습/검증 분할
train_x_, val_x, train_y_, val_y = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# MLP 분류기
import torch.nn.functional as F
import torch.nn as nn

from pytorch_tabnet.tab_model import TabNetClassifier

clf = TabNetClassifier(
    n_d=32, n_a=32, n_steps=5,
    gamma=1.5, n_independent=2, n_shared=2,
    cat_idxs=[], cat_dims=[], cat_emb_dim=[],
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=0.001),
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax', # "sparsemax" 도 가능
    verbose=1
)

clf.fit(
    X_train=train_x_, y_train=train_y_,
    eval_set=[(val_x, val_y)],
    eval_name=["val"],
    eval_metric=["accuracy"],
    max_epochs=200,
    patience=10,
    batch_size=32,
    virtual_batch_size=8,
    num_workers=0
)

# ░▒▓ 8. 테스트 예측 및 평가 ▓▒░
preds = clf.predict(X_test)

# 이진 라벨 다시 정리
y_test_bin = [0 if y == 0 else 1 for y in y_test]
preds_bin = [0 if y == 0 else 1 for y in preds]

print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test_bin, preds_bin))

print("\n📊 Classification Report:")
print(classification_report(y_test_bin, preds_bin, target_names=["Normal", "Abnormal"]))