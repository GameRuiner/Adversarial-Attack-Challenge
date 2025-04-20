from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn
import torch

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
    

def accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            labels = batch['identity'].to(device)

            logits, _ = model(images)
            print(logits.shape, labels)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            total_acc += accuracy(logits, labels)

    return total_loss / len(dataloader), total_acc / len(dataloader)


base_path = 'AdvCelebA'
# identity_file = os.path.join(base_path, 'identity_CelebA.txt')
# partition_file = os.path.join(base_path, 'list_eval_partition_no_overlap.txt')
# attack_status_file = os.path.join(base_path, 'attack_CelebA.txt')
# attack_info_file = os.path.join(base_path, 'final_attack_attackid_cw.txt')
# image_dir = os.path.join(base_path, 'images')

# train_loader, val_loader, test_loader, dataset = create_dataloaders_with_partition(
#         identity_file=identity_file,
#         partition_file=partition_file,
#         attack_status_file=attack_status_file,
#         attack_info_file=attack_info_file,
#         image_dir=image_dir,
#         batch_size=32
# )

# print(f"Number of identities: {dataset.num_identities}")
# print(f"Training samples: {len(train_loader.dataset)}")
# print(f"Validation samples: {len(val_loader.dataset)}")
# print(f"Testing samples: {len(test_loader.dataset)}")

# criterion = nn.CrossEntropyLoss()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# val_loss, val_acc = evaluate(model, val_loader, criterion, device)
# print(f"📈 Validation accuracy: {val_loss:.4f}")

# my_model = FaceRecognitionModel(num_classes=dataset.num_identities).to(device)

# val_loss, val_acc = evaluate(my_model, val_loader, criterion, device)
# print(f"📈 Validation accuracy: {val_loss:.4f}")