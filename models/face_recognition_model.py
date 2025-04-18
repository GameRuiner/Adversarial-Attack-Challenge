from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn

class FaceRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        weights = MobileNet_V2_Weights.DEFAULT
        base_model = mobilenet_v2(weights=weights)
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(base_model.last_channel, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        embedding = self.embedding(x)
        logits = self.classifier(embedding)
        return logits, embedding