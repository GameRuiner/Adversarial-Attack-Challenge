from torch.utils.data import Dataset
import os
from PIL import Image

class CelebADataset(Dataset):
    def __init__(self, label_file, image_dir, partition_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        self.partition_map = {}
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        with open(partition_file, 'r') as f:
            for line in f:
                img_name, partition = line.strip().split()
                base_name = os.path.splitext(img_name)[0]
                self.partition_map[base_name] = int(partition)

        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                self.samples.append((img_name, int(label)))

        for idx, (img_name, _) in enumerate(self.samples):
            base_name = os.path.splitext(img_name)[0]
            split = self.partition_map[base_name]
            if split == 0:
                self.train_indices.append(idx)
            elif split == 1:
                self.val_indices.append(idx)
            elif split == 2:
                self.test_indices.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label