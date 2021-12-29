import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from typing import Optional


def straitified_train_validation_split(embeddings, meta,targets, val_size=0.2):
    """
    Split dataset into train and validation sets.
    """
    # Get stratified split of data
    train_idx, val_idx = train_test_split(range(targets.shape[0]),
                                          test_size=val_size, 
                                          random_state=3, 
                                          stratify=targets)
    
    return {
            'train':(embeddings[train_idx], 
                    meta[train_idx], 
                    targets[train_idx]),
            'val': (embeddings[val_idx], 
                    meta[val_idx], 
                    targets[val_idx])
            }

def dataloader(embeddings, meta, targets, batch_size=32,
                num_workers=4, pin_memory=True):
    """
    Create dataloader for PyTorch.
    """
    return torch.utils.data.DataLoader(
        dataset=BillNetDataset(embeddings, meta, targets),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory)

class BillNetDataset(torch.utils.data.Dataset):
    """
    Dataset object to hold the data
    """
    def __init__(self, embeddings, meta, targets):
        self.embeddings = embeddings
        self.meta = meta
        self.targets = targets

    def __getitem__(self, index):
        embedding = self.embeddings[index]
        meta = self.meta[index]
        target = self.targets[index]
        
        return embedding, meta, target
    
    def __len__(self):
        return self.embeddings.shape[0]

class BillNetDataModule(pl.LightningDataModule):
    """
    
    """
    def __init__(self, batch_size=32,
                       data_path='data/processed/'):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path

    def setup(self):
        #Read tensors for training and val
        bert_train = torch.load(self.data_path + 'bert_train_103-114.pt')
        meta_train = torch.load(self.data_path + 'meta_train_103-114.pt')
        targets_train = torch.load(self.data_path + 'labels_train_103-114.pt')
        # Split training and val
        splits = straitified_train_validation_split(bert_train, meta_train, targets_train)
        self.train, self.val = splits['train'], splits['val']
        # read test
        self.test = (torch.load(self.data_path + 'bert_test_115.pt'), 
                    torch.load(self.data_path + 'meta_test_115.pt'), 
                    torch.load(self.data_path + 'labels_test_115.pt'))
   
    def train_dataloader(self):
        return dataloader(*self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return dataloader(*self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return dataloader(*self.test, batch_size=self.batch_size)