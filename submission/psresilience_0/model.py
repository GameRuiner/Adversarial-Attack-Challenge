import torch.nn as nn
import torch.nn.functional as F
import torch

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