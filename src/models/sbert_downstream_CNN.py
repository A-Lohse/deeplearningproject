import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import torchmetrics.functional as M
import numpy as np
from src.models.sbert_downstream_base import SBertDsBase

class SBertDsCNN(SBertDsBase):
    """
    Implements the Sentence Bert downstream model with CNN layers.
    args:
        include_meta: bool, whether to include meta data in the model.
        learning_rate: float, learning rate for the optimizer.
        dropout_rate: float, dropout rate for the model.
        class_weights: numpy array, class weights for the loss function.
    returns:
        model: pytorch model.
    """
    def __init__(self, include_meta:bool=False, class_weights=None, 
                 learning_rate=0.001, dropout_rate=0.0):
        super().__init__()

        self.learning_rate = learning_rate
        self.include_meta = include_meta
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        #define layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, 
                               kernel_size=3, stride=1, 
                               padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, 
                              kernel_size=3, stride=1, 
                              padding=0)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, 
                               kernel_size=3, stride=1, 
                               padding=0)
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout2d = nn.Dropout2d(p=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.LazyLinear(out_features = 120)         
        self.fc2 = nn.Linear(120, 2)
        
    def forward(self, x_emb, x_meta=None):
        x_emb = self.dropout2d(F.relu(self.batchnorm1(self.conv1(x_emb))))
        x_emb = self.pool(x_emb)
        x_emb = self.dropout2d(F.relu(self.batchnorm2(self.conv2(x_emb))))
        x_emb = self.pool(x_emb)
        x_emb = self.dropout2d(F.relu(self.conv3(x_emb)))
        x_emb = self.pool(x_emb)
        x_emb = torch.flatten(x_emb, 1)
        
        if self.include_meta or x_meta is not None:
            
            x = torch.cat([x_emb, x_meta], dim=1)
        else:
            x = x_emb
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x