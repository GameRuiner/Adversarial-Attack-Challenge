import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class Classifier:
    """Class to extract template data from face images"""

    def __init__(self):
        self.model_path = "cnn.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BinaryClassifier().to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def extract(self, image_path):
        """
        Extracts the classification score and decision from the image.
        Returns:
            score (float): Probability of attack.
            decision (bool): True if attack, False otherwise.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(image)
            prob = torch.sigmoid(logits).item()

        decision = prob > 0.5
        return prob, decision