import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import torchmetrics.functional as M
import numpy as np

class SBertDsBase(pl.LightningModule):
    """
    Implements a base class for SBERT based downstream models.
    """

    def calculate_metrics_helper(self, preds, targets, 
                                       step:str='val'):
        """
        Calculates metrics for a given set of predictions and targets.
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
        Helper function to get all preds and targets over an epoch
        and calculate metrics.
        """
        epoch_preds=[]
        epoch_targets=[]
        for out in epoch_outputs:
            epoch_preds=np.append(epoch_preds, out['preds'].cpu())
            epoch_targets=np.append(epoch_targets, out['targets'].cpu())
        return torch.tensor(epoch_preds, dtype=torch.long), torch.tensor(epoch_targets, dtype=torch.long)

    def common_step(self, batch):
        """
        Common step function for all the three phases of training.
        """
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
        self.log('train_loss', outputs['loss'], on_step=False, 
                 on_epoch=True, logger=False)
        return outputs
        # --------------------------
    def training_epoch_end(self, outputs) -> None:

        preds, targets = self.epoch_metrics(outputs)
        for k,v in self.calculate_metrics_helper(preds, targets, 
                                            step='train').items():
            self.log(k, v, logger=False)

    def validation_step(self, batch, batch_idx):
        outputs = self.common_step(batch)
        self.log('val_loss', outputs['loss'], on_step=False, 
                    on_epoch=True, logger=False)
        return outputs
    
    def validation_epoch_end(self, outputs) -> None:

        preds, targets = self.epoch_metrics(outputs)
        for k,v in self.calculate_metrics_helper(preds, targets, 
                                                 step='val').items():
            self.log(k, v, prog_bar=True, logger=False)
           
    
    def test_step(self, batch, batch_idx):
        outputs = self.common_step(batch)
        self.log('test_loss', outputs['loss'], on_step=False, 
                on_epoch=True, logger=False)
        return outputs

    def test_epoch_end(self, outputs) -> None:

        preds, targets = self.epoch_metrics(outputs)
        for k,v in self.calculate_metrics_helper(preds, targets, 
                                                step='test').items():
            self.log(k, v, logger=False)
    
    def configure_optimizers(self):
        """
        Implements the optimizer
        """
        return  torch.optim.Adam(self.parameters(), lr=self.learning_rate,
                                     weight_decay=0.01)