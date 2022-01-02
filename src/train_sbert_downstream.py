import pytorch_lightning as pl
import pandas as pd

#------------training callbacks---------#
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from src.callbacks.metrics_callback import MetricsCallback

#------------models and datamodule---------#
from src.models.sbert_downstream_CNN import SBertDsCNN
from src.models.sbert_downstream_FNN import SBertDsFNN
from src.data_modules.sbert_downstream_datamodule import SbertDSDataModule

from src.utils.utils import parse_metrics

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
        dm = SbertDSDataModule(weighted_sampler=True)
    else:
        dm = SbertDSDataModule(weighted_sampler=False)
    dm.setup()

    if criterion_class_weights == True:
        class_weights = dm.class_weights 
    else:
        class_weights = None
    #Define the models to run
    models = {
            #CNN models
            'CNN':SBertDsCNN(include_meta=False, class_weights=class_weights, 
                             learning_rate=lr, dropout_rate=dropout_rate),
            'CNN inc. meta':SBertDsCNN(include_meta=True, class_weights=class_weights,
                                        learning_rate=lr, dropout_rate=dropout_rate),
            #FNN models flattened sentence embeddings
            'FNN':SBertDsFNN(avg_emb=False, include_meta=False, class_weights=class_weights,
                              learning_rate=lr, dropout_rate=dropout_rate),
            'FNN inc. meta':SBertDsFNN(avg_emb=False, include_meta=True, class_weights=class_weights,
                                        learning_rate=lr, dropout_rate=dropout_rate),
            #FNN models avg. sentence embeddings
            'FNN avg':SBertDsFNN(avg_emb=True, include_meta=False, class_weights=class_weights,
                                  learning_rate=lr, dropout_rate=dropout_rate),
            'FNN avg inc. meta':SBertDsFNN(avg_emb=True, include_meta=True, class_weights=class_weights,
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