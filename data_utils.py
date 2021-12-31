import torch
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from typing import Optional
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data.sampler import WeightedRandomSampler

def straitified_train_validation_split(embeddings, meta,targets, val_size=0.2):
    """
    Split dataset into train and validation sets.
    """
    # Get stratified split of data
    train_idx, val_idx = train_test_split(range(targets.shape[0]),
                                          test_size=val_size, 
                                          random_state=1234, 
                                          stratify=targets)
    return {
            'train':(embeddings[train_idx], 
                    meta[train_idx], 
                    targets[train_idx]),
            'val': (embeddings[val_idx], 
                    meta[val_idx], 
                    targets[val_idx])
            }

def dataloader(embeddings, meta, targets, batch_size=16,
               num_workers=4, pin_memory=True, sampler=None):
    """
    Create dataloader for PyTorch.
    """
    return torch.utils.data.DataLoader(
        dataset=BillNetDataset(embeddings, meta, targets),
        batch_size=batch_size,
        sampler = sampler,
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
    
    def __init__(self, batch_size=16,
                       data_path='data/processed/'):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        

    def setup(self):
        # read tensors for training and val
        bert_train = torch.load(self.data_path + 'bert_train_103-114.pt')
        meta_train = torch.load(self.data_path + 'meta_train_103-114.pt')
        targets_train = torch.load(self.data_path + 'labels_train_103-114.pt')
        
        #get weights and define sampler
        self.weights = self.compute_sampling_weights(targets_train)
        #self.sampler = WeightedRandomSampler(weights, len(weights))                     
        # split training and val
        splits = straitified_train_validation_split(bert_train, meta_train, targets_train)
        self.train, self.val = splits['train'], splits['val']
        # read tensors for test
        self.test = (torch.load(self.data_path + 'bert_test_115.pt'), 
                     torch.load(self.data_path + 'meta_test_115.pt'), 
                     torch.load(self.data_path + 'labels_test_115.pt'))

    def compute_sampling_weights(self, train_targets):
        """Calculate class weights for sampler"""
        class_weights = compute_class_weight('balanced', 
                                             classes=np.unique(train_targets.numpy()),
                                             y=train_targets.numpy())
        return torch.tensor(class_weights, dtype=torch.float)
   
    def train_dataloader(self):
        return dataloader(*self.train, batch_size=self.batch_size)
                           
    def val_dataloader(self):
        return dataloader(*self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return dataloader(*self.test, batch_size=self.batch_size)
