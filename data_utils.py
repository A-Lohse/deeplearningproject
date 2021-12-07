import torch
from sklearn.model_selection import train_test_split
import numpy as np

data_path = 'data/processed/'

def straitified_train_validation_split(dataset, labels, val_size=0.2):
    """
    Split dataset into train and validation sets.
    """
    # Get stratified split of data
    train_idx, val_idx = train_test_split(range(labels.shape[0]), 
            test_size=val_size, random_state=3, stratify=labels)
    train_dataset, train_labels = dataset[train_idx], labels[train_idx]
    val_dataset, val_labels = dataset[val_idx], labels[val_idx]
    return train_dataset, train_labels, val_dataset, val_labels


def dataloader(dataset, labels, batch_size=32, shuffle=True,
                num_workers=4, pin_memory=True):
    """
    Create dataloader for PyTorch.
    """
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=Dataset(dataset, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)
    return dataloader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)
