import os
from PIL import Image
from torch.utils.data import Dataset


class FaceVerificationDataset(Dataset):
    def __init__(self, pairs_file, image_dir, transform=None):
        """
        Args:
            pairs_file (str): Path to file containing image pairs and labels.
            image_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.pairs = []

        with open(pairs_file, 'r') as f:
            for line in f:
                img1, img2, label = line.strip().split()
                self.pairs.append((img1, img2, int(label)))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_name, img2_name, label = self.pairs[idx]
        img1_path = os.path.join(self.image_dir, img1_name)
        img2_path = os.path.join(self.image_dir, img2_name)
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return {
            'image1': img1,
            'image2': img2,
            'label': label
        }