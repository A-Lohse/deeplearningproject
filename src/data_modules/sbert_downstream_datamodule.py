import torch
import pytorch_lightning as pl
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data.sampler import WeightedRandomSampler

class SbertDataset(torch.utils.data.Dataset):
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

class SbertDSDataModule(pl.LightningDataModule):

    """
    Datamodule for the sbert downstream model. Loads embeddings from 
    either the finetuned sentence-bert model or normal sentence-bert model.
    """
    
    def __init__(self, batch_size=16,
                weighted_sampler=True,
                data_path='data/processed/',
                ):
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.weighted_sampler = weighted_sampler

    def straitified_train_validation_split(self, embeddings, meta,targets, val_size=0.2):
        """
        Stratified split of the dataset into train and validation.
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
    
    def compute_weights(self, target, weight_type='class'):
        """
        Compute sample weights for the weighted random sampler 
        or class weights for the criterion function.
        """       
        if weight_type =='class':
            print('target train 0/1: {}/{}'.format(
            (target == 0).sum(), (target == 1).sum()))
            class_weights = compute_class_weight('balanced',
                                            classes=np.unique(target.numpy()),
                                            y=target.numpy())
            return torch.tensor(class_weights, dtype=torch.float)
        elif weight_type =='sample':
             # Compute samples weight (each sample should get its own weight)
            class_sample_count = torch.tensor(
                [(target == t).sum() for t in torch.unique(target, sorted=True)])
            weight = 1. / class_sample_count.float()
            return torch.tensor([weight[t] for t in target])

    def setup(self):
        # read tensors for training and val
        bert_train = torch.load(self.data_path + 'bert_train_103-114.pt')
        meta_train = torch.load(self.data_path + 'meta_train_103-114.pt')
        targets_train = torch.load(self.data_path + 'labels_train_103-114.pt')
                           
        # split training and val
        splits = self.straitified_train_validation_split(bert_train, meta_train, targets_train)
        self.train, self.val = splits['train'], splits['val']
         #get weights and define sampler
        sampling_weights = self.compute_weights(self.train[2], weight_type='sample')
        self.class_weights = self.compute_weights(self.train[2], weight_type='class')
        self.sampler = WeightedRandomSampler(sampling_weights, num_samples=len(sampling_weights)) 
        # read tensors for test
        self.test = (torch.load(self.data_path + 'bert_test_115.pt'), 
                     torch.load(self.data_path + 'meta_test_115.pt'), 
                     torch.load(self.data_path + 'labels_test_115.pt'))

    def train_dataloader(self):
        if self.weighted_sampler:
            return DataLoader(dataset = SbertDataset(*self.train), 
                              batch_size= self.batch_size,
                              sampler= self.sampler)
        else:
             return DataLoader(dataset = SbertDataset(*self.train), 
                               batch_size=self.batch_size)
                           
    def val_dataloader(self):
        return DataLoader(dataset = SbertDataset(*self.val),
                          batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(dataset=SbertDataset(*self.test), 
                          batch_size=self.batch_size)
