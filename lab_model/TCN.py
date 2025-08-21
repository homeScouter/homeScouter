
# 2. 필요한 라이브러리 임포트
import os
import shutil
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
from tqdm.auto import tqdm

# 이진 분류를 위한 라벨 맵
label_map = {"Normal": 0, "Violence": 1, "Weaponized": 2}

# 특징을 저장할 캐시 폴더 경로 설정
cache_folder = '/content/drive/MyDrive/cache_features'
os.makedirs(cache_folder, exist_ok=True)


# 4. 특징 추출 관련 함수들
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
        # 오류 발생 시 None 반환
        print(f"영상 로드 중 오류 발생: {e} - {video_path}")
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
        feat = feature_extractor(frames.to(device))

    # 모델의 출력을 1차원 텐서로 평탄화 (올바른 형태)
    return feat.view(-1).cpu()


# 5. 데이터셋 로드 함수
def load_dataset(root_folder, feature_extractor, cache_folder="cache_features"):
    """
    캐싱 기능을 포함하여 데이터셋을 로드하고 특징을 추출합니다.
    """
    xs, ys = [], []
    os.makedirs(cache_folder, exist_ok=True)

    for label_name, label_idx in label_map.items():
        folder = os.path.join(root_folder, label_name)
        if not os.path.isdir(folder):
            print(f"경로를 찾을 수 없음: {folder}. 이 폴더를 건너뜁니다.")
            continue

        video_files = [f for f in os.listdir(folder) if f.endswith(".avi")]
        print(f"{label_name}: {len(video_files)} videos")

        for vf in tqdm(video_files, desc=f"Processing {label_name} videos"):
            video_path = os.path.join(folder, vf)
            cache_path = os.path.join(cache_folder, f"{label_name}_{vf}.pt")

            if os.path.exists(cache_path):
                feature = torch.load(cache_path)
            else:
                frames = load_video_frames(video_path)
                if frames is not None:
                    feature = extract_feature(frames, feature_extractor, device)
                    if feature is not None:
                        torch.save(feature, cache_path)
                    else:
                        continue
                else:
                    continue

            xs.append(feature)
            ys.append(label_idx)

    # 특징이 None이 아닌 것들만 필터링
    valid_indices = [i for i, f in enumerate(xs) if f is not None]
    xs = [xs[i] for i in valid_indices]
    ys = [ys[i] for i in valid_indices]

    return xs, ys


# 6. MLP 분류기 모델
class ResidualMLP(nn.Module):
    def __init__(self, input_dim=512, num_classes=3, dropout_rate=0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.norm1 = nn.LayerNorm(64)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(64, 32)
        self.norm3 = nn.LayerNorm(32)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.out = nn.Linear(32, num_classes)  # num_classes를 3으로 설정

    def forward(self, x):
        # 입력 텐서의 차원 확인 및 reshape
        # x의 shape은 (batch_size, input_dim) 이어야 함

        x = F.relu(self.norm1(self.fc1(x)))
        x = self.dropout1(x)

        residual = x
        x = F.relu(self.norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = x + residual

        x = F.relu(self.norm3(self.fc3(x)))
        x = self.dropout3(x)

        return self.out(x)


# Dataset 클래스
class VideoFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.stack(features, dim=0).to(torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 7. 메인 학습 함수
def train_model():
    # Pre-trained 3D CNN 모델 준비
    print("모델 로딩 중...")
    model_3d = models.r2plus1d_18(weights=models.R2Plus1D_18_Weights.KINETICS400_V1)
    model_3d.eval()
    feature_extractor = torch.nn.Sequential(*list(model_3d.children())[:-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor.to(device)

    # 데이터셋 로딩
    dataset_folder = '/content/drive/MyDrive/Dataset'
    train_x, train_y = load_dataset(dataset_folder, feature_extractor, cache_folder)

    # 데이터 분할
    train_x, test_x, train_y, test_y = train_test_split(
        train_x, train_y, test_size=0.2, stratify=train_y, random_state=42
    )

    print(f"\nTrain samples: {len(train_x)}, Test samples: {len(test_x)}")

    # 훈련/검증 데이터 분할
    train_x_, val_x, train_y_, val_y = train_test_split(
        train_x, train_y, test_size=0.2, stratify=train_y, random_state=42
    )

    # MLP 분류기
    model_clf = ResidualMLP(num_classes=len(label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_clf.parameters(), lr=1e-6)

    # DataLoader 준비
    train_dataset = VideoFeatureDataset(train_x_, train_y_)
    val_dataset = VideoFeatureDataset(val_x, val_y)
    test_dataset = VideoFeatureDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    test_loader = DataLoader(test_dataset, batch_size=8)

    # 학습 루프
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None

    print("\n모델 학습 시작...")
    for epoch in tqdm(range(1000), desc="Training Epochs"):
        model_clf.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training", leave=False):
            x, y = x.to(device), y.to(device)
            out = model_clf(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        model_clf.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch + 1} Validation", leave=False):
                x, y = x.to(device), y.to(device)
                out = model_clf(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(all_labels, all_preds)

        tqdm.write(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model_clf.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                tqdm.write("Early stopping triggered.")
                break

    # Best 모델 로드 및 평가
    if best_model_state is not None:
        model_clf.load_state_dict(best_model_state)

    print("\n모델 테스트 평가 시작...")
    model_clf.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing Model"):
            x, y = x.to(device), y.to(device)
            out = model_clf(x)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print("\n🧾 Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\n📊 Classification Report:")
    target_names = list(label_map.keys())
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # 모델 저장
    os.makedirs('../camera_streamer', exist_ok=True)
    torch.save(model_clf.state_dict(), '../camera_streamer/best_residual_mlp_model.pth')
    print("\n모델이 'camera_streamer/best_residual_mlp_model.pth'에 저장되었습니다. ✅")


if __name__ == '__main__':
    train_model()