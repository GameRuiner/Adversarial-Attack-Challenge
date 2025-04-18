import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FacePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.pair_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)]
        self.transform = transform or transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pair_dirs)

    def __getitem__(self, idx):
        pair_dir = self.pair_dirs[idx]
        label = int(pair_dir.split('_')[-1])

        img_paths = sorted([os.path.join(pair_dir, fname) for fname in os.listdir(pair_dir)])
        img1 = Image.open(img_paths[0]).convert('RGB')
        img2 = Image.open(img_paths[1]).convert('RGB')

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return {
            'image1': img1,
            'image2': img2,
            'label': label
        }