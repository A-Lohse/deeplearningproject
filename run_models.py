from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Callback
import pytorch_lightning as pl
from models import BillNet_CNN, BillNet_FNN
from data_utils import BillNetDataModule
import pandas as pd
from collections import defaultdict
import copy 
import warnings
warnings.filterwarnings("ignore")

class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""
    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)


def parse_metrics(metrics:dict)->pd.DataFrame:
    """
    Helper to parse training/validation metrics from log
    """
    # remove metrics from sanity check
    results  = defaultdict(list)
    for res in metrics[1:]:
        #val metrics
        results['val_loss'].append(float(res['val_loss'].cpu()))
        results['val_acc'].append(float(res['val_acc'].cpu()))
        results['val_f1'].append(float(res['val_f1'].cpu()))
        results['val_precision'].append(float(res['val_precision'].cpu()))
        results['val_recall'].append(float(res['val_recall'].cpu()))
        results['val_prauc'].append(float(res['val_prauc'].cpu()))
        results['val_rocauc'].append(float(res['val_rocauc'].cpu()))
        #train metrics
        results['train_loss'].append(float(res['train_loss'].cpu()))
        results['train_acc'].append(float(res['train_acc'].cpu()))
        results['train_f1'].append(float(res['train_f1'].cpu()))
        results['train_precision'].append(float(res['train_precision'].cpu()))
        results['train_recall'].append(float(res['train_recall'].cpu()))
        results['train_prauc'].append(float(res['train_prauc'].cpu()))
        results['train_rocauc'].append(float(res['train_rocauc'].cpu()))
    return pd.DataFrame(results)


def run_models():
    #Set the seed
    pl.seed_everything(1234)
    #Load data module and extract class weights
    dm = BillNetDataModule()
    dm.setup()
    class_weights = dm.weights 
    #Define the models to run
    models = {
              #CNN models
              'CNN':BillNet_CNN(include_meta=False, 
                                class_weights=class_weights),
              'CNN inc. meta':BillNet_CNN(include_meta=True, 
                                          class_weights=class_weights),
              #FNN models flattened sentence embeddings
              'FNN':BillNet_FNN(avg_emb=False, include_meta=False, 
                                class_weights=class_weights),
              'FNN inc. meta':BillNet_FNN(avg_emb=False, include_meta=True,
                                          class_weights=class_weights),
              #FNN models avg. sentence embeddings
              'FNN avg':BillNet_FNN(avg_emb=True, include_meta=False,
                                    class_weights=class_weights),
              'FNN avg inc. meta':BillNet_FNN(avg_emb=True, include_meta=True,
                                              class_weights=class_weights)
              }
 
    #-------------------------#
    #train and test the models#
    #-------------------------#
    model_train_metrics = dict()
    test_metrics = dict()
    for model_name, model in models.items():
        print('-'*66)
        print(f'Training: {model_name}')
        print('-'*66)
        metrics_cb = MetricsCallback()
        checkpoint_cb = ModelCheckpoint(monitor='val_prauc',
                                        mode='max',
                                        save_top_k=1,
                                        dirpath='trained_models/',
                                        filename=model_name+"-{epoch:02d}-{val_loss:.2f}")
        early_stop_cb = EarlyStopping(monitor="val_prauc",
                                      patience=5,  
                                      mode="max")
        #setup trainer
        trainer = pl.Trainer(callbacks=[early_stop_cb, metrics_cb,
                                        checkpoint_cb], gpus=1,
                                        num_sanity_val_steps=0)
        #Fit model and find best lr
        #trainer.tune(model, train_dataloader=dm.train_dataloader())
        trainer.fit(model, datamodule=dm)

        # save model metrics
        model_train_metrics[model_name] = parse_metrics(metrics_cb.metrics)
        #Test the model
        trainer.test(datamodule=dm, ckpt_path='best')
        test_metrics[model_name] = trainer.logged_metrics
        #Save the model    

    print('Done!')
    pd.DataFrame(test_metrics).to_pickle('data/results/test_metrics.pkl')
    pd.to_pickle(model_train_metrics, 'data/results/train_val_losses.pkl')


if __name__ == '__main__':
    run_models()