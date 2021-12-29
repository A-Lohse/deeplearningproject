import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

class BillNet_CNN(pl.LightningModule):
    """
    Implements the CNN based model.
    """
    def __init__(self, include_meta:bool=False):
        super().__init__()
        self.include_meta = include_meta
        
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
        self.fc1 = nn.LazyLinear(out_features= 120)         
        self.fc2 = nn.Linear(120, 2)
        
    def forward(self, x_emb, x_meta):
        x_emb = F.relu(self.batchnorm1(self.conv1(x_emb)))
        x_emb = self.pool(x_emb)
        x_emb = F.relu(self.batchnorm2(self.conv2(x_emb)))
        x_emb = self.pool(x_emb)
        x_emb = F.relu(self.conv3(x_emb))
        x_emb = self.pool(x_emb)
        x_emb = torch.flatten(x_emb, 1)
        if self.include_meta:
            x = torch.cat([x_emb, x_meta], dim=1)
        else:
            x = x_emb
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        # --------------------------
        x_emb, target, x_meta = batch
        if self.include_meta:
            output = self(x_emb, x_meta)
        else:
            output = self(x_emb)
        loss = self.criterion(output, target)
        self.log('train_loss', loss)
        return loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        # --------------------------
        x_emb, target, x_meta = batch
        if self.include_meta:
            output = self(x_emb, x_meta)
        else:
            output = self(x_emb)
        loss = self.criterion(output, target)
        self.log('val_loss', loss)
        # --------------------------
    def test_step(self, batch, batch_idx):
        # --------------------------
        x_emb, target, x_meta = batch
        if self.include_meta:
            output = self(x_emb, x_meta)
        else:
            output = self(x_emb)
        loss = self.criterion(output, target)
        self.log('test_loss', loss)
        # --------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,
                                     weight_decay=0.01)
        return optimizer
    
    def criterion(self):
        return nn.CrossEntropyLoss()
        

class BillNet_FNN(pl.LightningModule):

    def __init__(self, num_classes:int=2, 
                 avg_emb:bool=False,
                 include_meta:bool=False):
        super().__init__()
        self.include_meta = include_meta
        self.avg_emb = avg_emb
        #------#
        #Layers#
        #------#
        self.fc1 = nn.LazyLinear(out_features=256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        #self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x_emb, x_meta):
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
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)        
        return x

    def training_step(self, batch, batch_idx):
        # --------------------------
        x_emb, x_meta, target = batch
        if self.include_meta:
            output = self(x_emb.float(), x_meta.float())
        else:
            output = self(x_emb.float())
        criterion = self.criterion()
        loss = criterion(output, target)
        self.log('train_loss', loss)
        return loss
        # --------------------------

    def validation_step(self, batch, batch_idx):
        # --------------------------
        x_emb, x_meta, target = batch
        if self.include_meta:
            output = self(x_emb.float(), x_meta.float())
        else:
            output = self(x_emb.float())
        criterion = self.criterion()    
        loss = criterion(output, target)
        self.log('val_loss', loss)
        # --------------------------
    
    def test_step(self, batch, batch_idx):
        x_emb, x_meta, target = batch
        if self.include_meta:
            output = self(x_emb.float(), x_meta.float())
        else:
            output = self(x_emb.float())
        criterion = self.criterion()    
        loss = criterion(output, target)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,
                                     weight_decay=0.01)
        return optimizer
    
    def criterion(self):
        return nn.CrossEntropyLoss()