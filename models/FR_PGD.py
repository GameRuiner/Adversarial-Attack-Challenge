import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import random
import re

# ==============================
# Device Setup
# ==============================
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# Haar Cascade Face Detection
# ==============================
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

def detect_face_haar(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (112, 112))
        return face
    else:
        return np.zeros((112, 112, 3), dtype=np.uint8)

# ==============================
# SiameseFaceDataset for Contrastive Loss
# ==============================
class SiameseFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.pairs = []
        self.transform = transform

        # Walk through all subdirectories
        for root, _, _ in os.walk(root_dir):
            match = re.search(r'label_(\d+)', root)
            if not match:
                continue
            label = int(match.group(1))
            im0 = next((f for f in os.listdir(root) if f.startswith("im_0_")), None)
            im1 = next((f for f in os.listdir(root) if f.startswith("im_1_")), None)
            if im0 and im1:
                img0_path = os.path.join(root, im0)
                img1_path = os.path.join(root, im1)
                self.pairs.append((img0_path, img1_path, label))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        img0_path, img1_path, label = self.pairs[idx]
        img0 = cv2.imread(img0_path)
        img1 = cv2.imread(img1_path)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img0 = detect_face_haar(img0)
        img1 = detect_face_haar(img1)
        if self.transform:
            img0 = self.transform(image=img0)['image']
            img1 = self.transform(image=img1)['image']
        return img0.to(torch.float32), img1.to(torch.float32), torch.tensor(label, dtype=torch.float32)
        
# ==============================
# Transforms
# ==============================
transform = A.Compose([
    A.Resize(112, 112),
    A.Normalize(),
    ToTensorV2()
])

# ==============================
# Siamese Network (Shared backbone)
# ==============================
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        with torch.no_grad():
            dummy = self.backbone(torch.zeros(1, 3, 112, 112))
            self.flattened = dummy.view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(self.flattened, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return F.normalize(self.fc(x), p=2, dim=1)  # L2 normalize

    def forward(self, img1, img2):
        return self.forward_once(img1), self.forward_once(img2)

# ==============================
# PGD Attack for Pairs
# ==============================
def pgd_attack(model, img1, img2, label, criterion, eps=0.03, alpha=0.01, iters=10):
    img1_adv = img1.clone().detach().to(device)
    img1_orig = img1.clone().detach().to(device)
    img1_adv.requires_grad = True
    for _ in range(iters):
        out1, out2 = model(img1_adv, img2)
        loss = criterion(out1, out2, label)
        model.zero_grad()
        if img1_adv.grad is not None:
            img1_adv.grad.zero_()
        loss.backward()
        img1_adv = img1_adv + alpha * img1_adv.grad.sign()
        eta = torch.clamp(img1_adv - img1_orig, min=-eps, max=eps)
        img1_adv = torch.clamp(img1_orig + eta, 0, 1).detach()
        img1_adv.requires_grad = True
    return img1_adv

# ==============================
# Contrastive Loss
# ==============================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dists = F.pairwise_distance(output1, output2, p=2)
        loss = label * dists.pow(2) + (1 - label) * F.relu(self.margin - dists).pow(2)
        return loss.mean()

# ==============================
# Dataset, Model, Loss, Optimizer
# ==============================
base_path = './AdvLFW/images'  # Update as needed
dataset = SiameseFaceDataset(base_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

model = SiameseNetwork().to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ==============================
# Training Loop (with PGD)
# ==============================
def train(model, loader, criterion, optimizer, epochs=10, use_pgd=True, save_path='models_bin/fr_pgd.pth'):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for img1, img2, label in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            if use_pgd:
                img1_adv = pgd_attack(model, img1, img2, label, criterion, eps=0.03, alpha=0.01, iters=7)
                out1, out2 = model(img1_adv, img2)
            else:
                out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {total_loss / len(loader):.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ==============================
# Evaluation: Cosine Similarity
# ==============================
def evaluate(model, loader, threshold=0.7):  # threshold closer to 1 for cosine similarity
    model.eval()
    y_true, y_pred, sims = [], [], []
    with torch.no_grad():
        for img1, img2, label in loader:
            img1, img2 = img1.to(device), img2.to(device)
            out1, out2 = model(img1, img2)
            sim = F.cosine_similarity(out1, out2).item()
            pred = 1 if sim > threshold else 0
            y_true.append(int(label.item()))
            y_pred.append(pred)
            sims.append(sim)
    acc = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, sims)
    except:
        roc_auc = None
    print("\nEvaluation (Cosine Similarity):")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}" if roc_auc else "ROC AUC: N/A")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# ==============================
# Run Training and Evaluation
# ==============================
train(model, train_loader, criterion, optimizer, epochs=10, use_pgd=True)
evaluate(model, test_loader, threshold=0.819)
