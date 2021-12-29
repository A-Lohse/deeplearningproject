from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from models_lightning import BillNet_CNN, BillNet_FNN
from data_utils import BillNetDataModule


def BillNet_main():
    #Set the seed
    pl.seed_everything(1234)
    #Load data module
    dm = BillNetDataModule()
    dm.setup()
    #----------------#
    #Train the models#
    #----------------#
    #TODO: change monitoring to F1 score?
    early_stop_callback = EarlyStopping(monitor="val_loss", 
                                        min_delta=0.00, 
                                        patience=3, 
                                        verbose=False, 
                                        mode="max")
    trainer = pl.Trainer(callbacks=[early_stop_callback],
                         gpus=1)
    #Define models to run
    models = [
              BillNet_CNN(include_meta=True),
              BillNet_FNN(avg_emb=False, include_meta=True),
              BillNet_FNN(avg_emb=True, include_meta=True)
              ]
    
    for model in models:
        #Fit model
        trainer.fit(model, datamodule=dm)
        #TODO
        #Test the model
        
if __name__ == '__main__':
    
    BillNet_main()