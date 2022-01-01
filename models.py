import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import torchmetrics.functional as M
import numpy as np

class BillNetBase(pl.LightningModule):
    """
    Implements BillNet base model.
    """

    def calculate_metrics_helper(self, preds, targets, 
                                       step:str='val'):
        """
        Calculates metrics
        """
        return {
                step+'_acc': M.accuracy(preds, targets),
                step+'_f1': M.f1(preds, targets),
                step+'_precision':M.precision(preds, targets),
                step+'_recall':M.recall(preds, targets),
                step+'_prauc': M.average_precision(preds, targets),
                step+'_rocauc': M.auroc(preds, targets)
                }

    def epoch_metrics(self, epoch_outputs):
        """
        Calculate metrics over an epoch
        """
        epoch_preds=[]
        epoch_targets=[]
        for out in epoch_outputs:
            epoch_preds=np.append(epoch_preds, out['preds'].cpu())
            epoch_targets=np.append(epoch_targets, out['targets'].cpu())
        return torch.tensor(epoch_preds, dtype=torch.long), torch.tensor(epoch_targets, dtype=torch.long)

    def common_step(self, batch):
        x_emb, x_meta, targets = batch
        if self.include_meta:
            output = self(x_emb.float(), x_meta.float())
        else:
            output = self(x_emb.float())
        loss = self.criterion(output, targets)
        preds = output.argmax(dim=1)
    
        return {'loss':loss, 'output':output, 
                'targets':targets, 'preds':preds}

    def training_step(self, batch, batch_idx):
        # --------------------------
        outputs = self.common_step(batch)
        self.log('train_loss', outputs['loss'], on_step=False, on_epoch=True)
        return outputs
        # --------------------------
    def training_epoch_end(self, outputs) -> None:

        preds, targets = self.epoch_metrics(outputs)
        for k,v in self.calculate_metrics_helper(preds, targets, 
                                            step='train').items():
            self.log(k, v)

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch)
        self.log('val_loss', outputs['loss'], on_step=False, on_epoch=True)
        return outputs
    
    def validation_epoch_end(self, outputs) -> None:

        preds, targets = self.epoch_metrics(outputs)
        for k,v in self.calculate_metrics_helper(preds, targets, 
                                                 step='val').items():
            self.log(k, v, prog_bar=True)
           
    
    def test_step(self, batch, batch_idx):
        outputs = self.common_step(batch)
        self.log('test_loss', outputs['loss'], on_step=False, on_epoch=True)
        return outputs

    def test_epoch_end(self, outputs) -> None:

        preds, targets = self.epoch_metrics(outputs)
        for k,v in self.calculate_metrics_helper(preds, targets, 
                                                step='test').items():
            self.log(k, v)
    
    def configure_optimizers(self):
        return  torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=0.01)

class BillNet_CNN(BillNetBase):
    """
    Implements the BillNet CNN model.
    Class weights should only be used in the criterion 
    if weighted random sampler is not used.
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
        if self.include_meta:
            x = torch.cat([x_emb, x_meta], dim=1)
        else:
            x = x_emb
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class BillNet_FNN(BillNetBase):
    """
    BillNet feed-forward architecture.
    Args:

    Class weights should only be used in the criterion 
    if weighted random sampler is not used.
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