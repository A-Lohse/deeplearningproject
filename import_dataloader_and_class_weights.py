from typing import Union
import os
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd
from data_utils import straitified_train_validation_split, dataloader


def identifyIfInstanceOnColab() -> bool:
    """This function identifies, whether the current ipynb instance is ran on colab.

    Returns:
        bool: True if instance is on colab
    """
    if 'google.colab' in str(get_ipython()):
        return True
    else:
        return False


def loadColab(path_text: str, path_label: str) -> Union[any, any]:
    """Loads text and labels for bills from google colab

    Args:
        path_text (str): path for training data, bill texts
        path_label (str): path for training data, bill labels

    Returns:
        Union[any,any]: [Unpickled collection of document texts, unpickled collection of document labels] 
    """
    # Read tensors
    bert_train = torch.load(path_text)
    labels_train = torch.load(path_label)
    return bert_train, labels_train


def loadLocal(path_text: str, path_label: str, file_id_text: str, file_id_label: str) -> Union[any, any]:
    """Loads text and labels for bills from google colab

    Args:
        path_text (str): path for text data
        path_label (str): path for label data
        file_id_text (str): Colab file id to download , if file does not exist locally
        file_id_label (str): Colab file id to download, if file does not exist locally

    Returns:
        Union[any,any]: [Unpickled collection of document texts, unpickled collection of document labels] 
    """

    if not os.path.exists(path_text):
        gdd.download_file_from_google_drive(file_id=file_id_text,
                                            dest_path=path_text,
                                            unzip=True)
    if not os.path.exists(path_label):
        gdd.download_file_from_google_drive(file_id=file_id_label,
                                            dest_path=path_label,
                                            unzip=True)
    bert_train = torch.load(path_text)
    labels_train = torch.load(path_label)
    return bert_train, labels_train


def import_data_loaders_and_class_weights(
    path_text_directory='./data/bert_train_103-114.pt',
    path_label_directory='./data/labels_train_103-114.pt',
    colab_text_file_id='1IZPbb7KEc9fjAItPf1JrPw5Je1Q0woNa',
    colab_label_file_id='1hvyLUS6e9tI_ICietdKUy4XCwurr6lSU'
) -> Union[dataloader, dataloader, torch.tensor]:
    """1. Loads training and validation data 
       2. Splits data using stratified validation split
       3. Calculate class weights for loss function
       4. Generate dataloader for a training and validation split

    Returns:
        Union[dataloader, dataloader, torch.tensor]: [dataloader_training, 
                                                      dataloader_validation, 
                                                      tensor with class weights]
    """
    if identifyIfInstanceOnColab():
        bert_train, labels_train = loadColab('data/processed/bert_train_103-114.pt',
                                             'data/processed/labels_train_103-114.pt')
    else:
        bert_train, labels_train = loadLocal(path_text_directory,
                                             path_label_directory,
                                             colab_text_file_id,
                                             colab_label_file_id)

    # Stratified split into training and validation
    bert_train, labels_train, bert_val, labels_val = straitified_train_validation_split(bert_train,
                                                                                        labels_train)

    # Calculate class weights for loss function because of unbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(labels_train.numpy()),
                                         y=labels_train.numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    # Load into training and validation dataloaders
    train_dataloader = dataloader(bert_train, labels_train, batch_size=64)
    valid_dataloader = dataloader(bert_val, labels_val, batch_size=64)

    # delete to free memory
    del bert_train, bert_val, labels_train, labels_val

    # Return output
    return train_dataloader, valid_dataloader, class_weights
