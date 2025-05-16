import os
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
import pandas as pd


class ResilienceDataset(Dataset):
    def __init__(self, identity_file_path, attack_status_file_path, attack_info_file_path, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        self.identity_df = pd.read_csv(identity_file_path, sep=' ', header=None,
                                     names=['image_name', 'identity'])
        self.unique_identities = self.identity_df['identity'].unique()
        self.identity_to_label = {identity: idx for idx, identity in enumerate(self.unique_identities)}


        self.attack_status_df = pd.read_csv(attack_status_file_path, sep=' ', header=None,
                                          names=['image_name', 'is_attacked'])

        self.attack_df = pd.read_csv(attack_info_file_path, sep=' ', header=None,
                                        names=['attacked_image', 'original_identity', 
                                               'attack_type', 'target_image', 'attack_id'])
        self.attack_info_map = {}
        for _, row in self.attack_df.iterrows():
                self.attack_info_map[row['attacked_image']] = {
                    'attack_type': row['attack_type'],
                    'target_image': row['target_image'],
                    'attack_id': row['attack_id']
                }

        merged_df = pd.merge(self.identity_df, self.attack_status_df, on='image_name', how='inner')
        self.combined_data = []

        for _, row in merged_df.iterrows():
            sample = {
                'image_name': row['image_name'],
                'identity': row['identity'],
                'is_attacked': row['is_attacked'] == 1
            }
        
            if sample['is_attacked'] and row['image_name'] in self.attack_info_map:
                attack_details = self.attack_info_map[row['image_name']]
                sample['attack_type'] = attack_details['attack_type']
                sample['target_image'] = attack_details['target_image']
                sample['attack_id'] = attack_details['attack_id']
                if attack_details['attack_type'] == 'impersonation':
                    target_img = attack_details['target_image']
                    target_matches = self.identity_df[self.identity_df['image_name'] == target_img]
                    if not target_matches.empty:
                        sample['target_identity'] = target_matches.iloc[0]['identity']
                    else:
                        sample['target_identity'] = -1
            else:
                sample['attack_type'] = "none"
                sample['target_image'] = ""
                sample['target_identity'] = -1
                sample['attack_id'] = -1
            
            self.combined_data.append(sample)
        
        self.num_identities = len(self.unique_identities)


    def __len__(self):
        return len(self.combined_data)

    def __getitem__(self, idx):
        sample = self.combined_data[idx]
        img_path = os.path.join(self.image_dir, sample['image_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        identity_label = self.identity_to_label[sample['identity']]
        target_label = -1
        if sample['is_attacked'] and sample['attack_type'] == 'impersonation' and sample['target_identity'] is not None:
            target_label = self.identity_to_label[sample['target_identity']]
        
        result = {
            'image': image,
            'identity': identity_label,
            'is_attacked': 1 if sample['is_attacked'] else 0,
            'attack_type': sample['attack_type'],
            'target_identity': target_label,
            'attack_id': sample['attack_id'],
            'image_name': sample['image_name']
        }
        
        return result


def create_dataloaders_with_partition(identity_file, partition_file, attack_status_file,
                                     attack_info_file=None, image_dir='images',
                                     batch_size=32):
    transform = transforms.ToTensor()

    train_dataset = ResilienceDataset(
        identity_file_path=identity_file,
        attack_status_file_path=attack_status_file,
        attack_info_file_path=attack_info_file,
        image_dir=image_dir,
        transform=transform
    )

    partition_df = pd.read_csv(partition_file, sep=' ', header=None,
                              names=['image_name', 'partition'])
    
    train_indices = []
    val_indices = []
    test_indices = []

    image_to_idx = {}
    for idx, item in enumerate(train_dataset.combined_data):
        image_to_idx[item['image_name']] = idx
    
    for _, row in partition_df.iterrows():
        img_name = row['image_name']
        partition = row['partition']
        
        if img_name in image_to_idx:
            idx = image_to_idx[img_name]
            if partition == 0:
                train_indices.append(idx)
            elif partition == 1:
                val_indices.append(idx)
            elif partition == 2:
                test_indices.append(idx)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    test_subset = Subset(train_dataset, test_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False
    )
    
    test_loader = DataLoader(
        test_subset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader, train_dataset