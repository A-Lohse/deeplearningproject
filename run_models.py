from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from models import BillNet_CNN, BillNet_FFNN
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

if __name__ == '__main__':
    BillNet_main()