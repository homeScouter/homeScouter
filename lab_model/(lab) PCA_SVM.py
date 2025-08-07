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

# train_x가 list of torch.Tensor 라면 numpy 배열로 변환 필요
X_train_np = torch.stack(train_x).numpy()
X_test_np = torch.stack(test_x).numpy()

# PCA 객체 생성 및 학습
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_np)
X_test_pca = pca.transform(X_test_np)

def convert_to_binary(labels):
    return [0 if y == 0 else 1 for y in labels]

train_y_bin = convert_to_binary(train_y)
test_y_bin = convert_to_binary(test_y)

train_x_, val_x, train_y_, val_y = train_test_split(
    X_train_pca, train_y_bin, test_size=0.2, stratify=train_y_bin, random_state=42
)


val_y = convert_to_binary(val_y)

clf = SVC(kernel='rbf', C=1.0, class_weight='balanced')  # class imbalance 대응
clf.fit(X_train_pca, train_y_bin)

y_pred_bin = clf.predict(X_test_pca)

print("📊 Confusion Matrix:")
print(confusion_matrix(test_y_bin, y_pred_bin))
print("📄 Classification Report:")
print(classification_report(test_y_bin, y_pred_bin, target_names=["Normal", "Abnormal"]))