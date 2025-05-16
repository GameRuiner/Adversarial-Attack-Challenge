import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re

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

def create_dataloaders_with_partition(image_dir='images', batch_size=32):
    transform = A.Compose([
        A.Resize(112, 112),
        A.Normalize(),
        ToTensorV2()
    ])
    dataset = SiameseFaceDataset(image_dir, transform=transform)
    train_ratio = 0.7
    val_ratio = 0.15
    total_size = len(dataset)
    print(total_size)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader, dataset