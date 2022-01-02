from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Callback
import pytorch_lightning as pl
from models import BillNet_CNN, BillNet_FNN
from src.data_modules.data_utils import BillNetDataModule
import pandas as pd
from collections import defaultdict
import copy 
#remove experimental warning from lazylayer
import warnings
warnings.filterwarnings("ignore")

#-------------#
# HYPERPARAMS #
#-------------#

#If random samples or criterion should be weighted
weighted_sampler = True
criterion_class_weights = False
auto_lr = False #Use lightning to find optimal lr
lr = 1e-3
dropout_rate = 0.2
monitor_metric = 'val_prauc' #area under the precision-recall curve
#-----------------#


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
    print('Starting to train models')
    pl.seed_everything(1234)
    print('HYPERPARAMS:')
    print('-'*33)
    print(f'Dropout rate: {dropout_rate}')
    print(f'Learning rate: {lr}')
    print(f'Early stopping monitor metric: {monitor_metric}')
    print(f'Weighted samples: {weighted_sampler}')
    print(f'Weighted classes in criterion: {criterion_class_weights}')
    print('-'*33)
    #Load and setup data
    if weighted_sampler:
        dm = BillNetDataModule(weighted_sampler=True)
    else:
        dm = BillNetDataModule(weighted_sampler=False)
    dm.setup()

    if criterion_class_weights == True:
        class_weights = dm.class_weights 
    else:
        class_weights = None
    #Define the models to run
    models = {
            #CNN models
            'CNN':BillNet_CNN(include_meta=False, class_weights=class_weights, 
                             learning_rate=lr, dropout_rate=dropout_rate),
            'CNN inc. meta':BillNet_CNN(include_meta=True, class_weights=class_weights,
                                        learning_rate=lr, dropout_rate=dropout_rate),
            #FNN models flattened sentence embeddings
            'FNN':BillNet_FNN(avg_emb=False, include_meta=False, class_weights=class_weights,
                              learning_rate=lr, dropout_rate=dropout_rate),
            'FNN inc. meta':BillNet_FNN(avg_emb=False, include_meta=True, class_weights=class_weights,
                                        learning_rate=lr, dropout_rate=dropout_rate),
            #FNN models avg. sentence embeddings
            'FNN avg':BillNet_FNN(avg_emb=True, include_meta=False, class_weights=class_weights,
                                  learning_rate=lr, dropout_rate=dropout_rate),
            'FNN avg inc. meta':BillNet_FNN(avg_emb=True, include_meta=True, class_weights=class_weights,
                                            learning_rate=lr, dropout_rate=dropout_rate)
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
        checkpoint_cb = ModelCheckpoint(monitor=monitor_metric,
                                        mode='max',
                                        save_top_k=1,
                                        dirpath='trained_models/',
                                        filename=model_name+"-{epoch:02d}-{val_loss:.2f}")
        early_stop_cb = EarlyStopping(monitor=monitor_metric,
                                      patience=5,  
                                      mode="max")
        #setup trainer
        trainer = pl.Trainer(callbacks=[early_stop_cb, metrics_cb,
                                        checkpoint_cb], 
                             gpus=1, num_sanity_val_steps=0,
                             auto_lr_find=auto_lr)
        
        if auto_lr:
            #Initialize parameters of lazy layer
            # for tuner to work
            batch = next(iter(dm.train_dataloader()))
            model.train()
            model.forward(batch[0].float(), batch[1].float())
            #find best learning rate
            trainer.tune(model, train_dataloader=dm.train_dataloader())
        
        #fit model
        trainer.fit(model, datamodule=dm)
        # save model metrics
        model_train_metrics[model_name] = parse_metrics(metrics_cb.metrics)
        #Test the model
        trainer.test(datamodule=dm, ckpt_path='best')
        test_metrics[model_name] = trainer.logged_metrics
        #Save the model    

    print('Done!')
    #save files
    pd.DataFrame(test_metrics).to_pickle('data/results/test_metrics.pkl')
    pd.to_pickle(model_train_metrics, 'data/results/train_val_metrics.pkl')

if __name__ == '__main__':
    run_models()