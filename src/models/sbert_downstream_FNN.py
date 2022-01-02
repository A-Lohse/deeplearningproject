import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import torchmetrics.functional as M
import numpy as np
from sbert_downstream_base import SBertDsBase

class SBertDsFNN(SBertDsBase):
    """
    Implements the Sentence Bert downstream model with FNN layers.
    Args:
        include_meta: bool, whether to include meta data in the model.
        learning_rate: float, learning rate for the optimizer.
        dropout_rate: float, dropout rate for the model.
        class_weights: numpy array, class weights for the loss function.
        avg_emb: bool, whether to average the embeddings of the sentences in each document.
    returns:
        model: pytorch model.
    """

    def __init__(self, 
                 avg_emb:bool=False,
                 include_meta:bool=False,
                 class_weights=None,
                 learning_rate=0.001,
                 dropout_rate=0.0):
        super().__init__()
        self.include_meta = include_meta
        self.avg_emb = avg_emb
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.learning_rate = learning_rate
       
        #define layers
        self.fc1 = nn.LazyLinear(out_features=256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_emb, x_meta=None):
        if self.avg_emb:
            #take the mean of the sentences in each document (dim 2 (with 0 indexing))
            x_emb = torch.mean(x_emb,axis = 2) 
            x_emb = torch.squeeze(x_emb)
        else:
            x_emb = torch.flatten(x_emb, 1)
        
        if self.include_meta:
            x = torch.cat([x_emb, x_meta], dim=1)
        else:
            x = x_emb
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)        
        return x