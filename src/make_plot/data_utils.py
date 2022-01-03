import torch
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix, roc_auc_score, average_precision_score
import pandas as pd

data_path = 'data/processed/'

def straitified_train_validation_split(dataset, labels, val_size=0.2, idx = False):
    """
    Split dataset into train and validation sets.
    """
    # Get stratified split of data
    train_idx, val_idx = train_test_split(range(labels.shape[0]), 
            test_size=val_size, random_state=3, stratify=labels)
    train_dataset, train_labels = dataset[train_idx], labels[train_idx]
    val_dataset, val_labels = dataset[val_idx], labels[val_idx]

    if idx:
        return train_dataset, train_labels, val_dataset, val_labels, train_idx, val_idx
    else:
    	return train_dataset, train_labels, val_dataset, val_labels


def dataloader(dataset, labels, batch_size=32, shuffle=True,
                num_workers=4, pin_memory=True):
    """
    Create dataloader for PyTorch.
    """
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=Dataset(dataset, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory)
    return dataloader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y
    
    def __len__(self):
        return len(self.data)

def metrics(labels_val, val_preds, test_labels, test_preds):
    print('-----------Validation Metrics-----------')
    print('Accuracy: {}'.format(accuracy_score(labels_val, val_preds)))
    print('F1: {}'.format(f1_score(labels_val, val_preds)))
    print('Precision: {}'.format(precision_score(labels_val, val_preds)))
    print('Recall: {}'.format(recall_score(labels_val, val_preds)))
    print('Confusion Matrix:')
    print(confusion_matrix(labels_val, val_preds))
    
    
    print('-----------Test Metrics-----------')
    print('Accuracy: {}'.format(accuracy_score(test_labels, test_preds)))
    print('F1: {}'.format(f1_score(test_labels, test_preds)))
    print('Precision: {}'.format(precision_score(test_labels, test_preds)))
    print('Recall: {}'.format(recall_score(test_labels, test_preds)))
    print('Confusion Matrix:')
    print(confusion_matrix(test_labels, test_preds))
    
    print("Valudation AUC:",roc_auc_score(labels_val, val_preds))
    print("Test AUC:", roc_auc_score(test_labels, test_preds))
    print("Valudation Avg. PR:",average_precision_score(labels_val, val_preds))
    print("Test Avg. PR:", average_precision_score(test_labels, test_preds))
    
def tex_table(labels_val, val_preds, test_labels, test_preds, name = None, path = None):
    v1 = round(accuracy_score(labels_val, val_preds),2)
    v2 = round(f1_score(labels_val, val_preds),2)
    v3 = round(precision_score(labels_val, val_preds),2)
    v4 = round(recall_score(labels_val, val_preds),2)
    v5 = round(roc_auc_score(labels_val, val_preds),2)
    v6 = round(average_precision_score(labels_val, val_preds),2)
    
    
    t1 = round(accuracy_score(test_labels, test_preds),2)
    t2 = round(f1_score(test_labels, test_preds),2)
    t3 = round(precision_score(test_labels, test_preds),2)
    t4 = round(recall_score(test_labels, test_preds),2)
    t5 = round(roc_auc_score(test_labels, test_preds),2)
    t6 = round(average_precision_score(test_labels, test_preds),2)
    
    val = [v1,v2,v3,v4,v5,v6]
    test = [t1,t2,t3,t4,t5,t6]
    if name is None:
        cat1 = 'Val'
        cat2 = 'Test'
    else:
        cat1 = name + " Val"
        cat2 = name + " Test"
        
    d = {cat1 : val,
         cat2 : test}
    tab = pd.DataFrame(d)
    tab = tab.T
    tab.columns = ["Acc.","F1","Pr,","Recall","AUC","Avg. pr"]
    
    if path == None:
        print("No name provided - saved as table.tex to dir")
        fn = "table.tex"
    else:    
        fn = path + ".tex"
    with open(fn, 'w') as f:
        f.write(tab.to_latex(index = True))
    
    return tab