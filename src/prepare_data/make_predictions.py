import os

os.chdir("C:\\Users\\tnv664\\OneDrive - University of Copenhagen\\Documents\\Uni\\Ph.D\\deep learning\\deeplearningproject")

from src.make_plot.data_utils import straitified_train_validation_split, dataloader, metrics

from src.models.sbert_downstream_CNN import SBertDsCNN
from src.models.sbert_downstream_FNN import SBertDsFNN
from src.data_modules import sbert_downstream_datamodule
import torch
import torch.nn as nn
import pickle

model_path = "src\\models\\trained_models\\"
data_path = 'data\\'

model_dict = {}

for m in os.listdir(model_path):
    if "baseline" in m: #do not do anything to the baseline models 
        pass
    else:    
        model_dict[m] = {}    
        model_dict[m]['val'] = {}
        model_dict[m]['test'] = {}
        
        #set up the dataloader and model
        if 'CNN' in m:
            if 'meta' in m:
                my_model = SBertDsCNN(include_meta = True)
            else:
                my_model = SBertDsCNN(include_meta = False)
        else: #else FNN
            if 'meta' in m:
                my_model = SBertDsFNN(include_meta = True)
            else:
                my_model = SBertDsFNN(include_meta = False)
                
        model = m
        dm = sbert_downstream_datamodule.SbertDSDataModule(data_path = data_path)
        dm.setup()
    
        #make the net
        net= my_model.load_from_checkpoint(model_path + model, strict = False)
        
        net.eval()
        val_preds, val_targs, val_probas = [], [], []
        test_preds, test_targs, test_probas = [], [], []
    
        with torch.no_grad():
            for local_batch, local_meta, local_labels in dm.val_dataloader():
                if "FNN" in m:
                    if 'avg' in m:
                        local_batch = torch.mean(local_batch,axis = 2) #take the mean of the sentences in each document (dim 2 (with 0 indexing))
                        local_batch = torch.squeeze(local_batch) #remove the channel dim we put there for CNN
                    else:
                        local_batch.flatten(start_dim = 1) #concat the the input
        
                outputs = net(local_batch.float(), local_meta)
                predicted = torch.max(outputs.data, 1)[1]
                val_probas += list(outputs.data.numpy()[:,1])
                val_targs += list(local_labels.numpy())
                val_preds += list(predicted.data.numpy())
            ####TEST DATA####
            for local_batch, local_meta, local_labels in dm.test_dataloader():
                if "FNN" in m:
                    if 'avg' in m:
                        local_batch = torch.mean(local_batch,axis = 2) #take the mean of the sentences in each document (dim 2 (with 0 indexing))
                        local_batch = torch.squeeze(local_batch) #remove the channel dim we put there for CNN
                    else:
                        local_batch.flatten(start_dim = 1) #concat the the input
                        
                outputs = net(local_batch.float(), local_meta)
                predicted = torch.max(outputs.data, 1)[1]
                test_probas += list(outputs.data.numpy()[:,1])
                test_targs += list(local_labels.numpy())
                test_preds += list(predicted.data.numpy())
        
        
        model_dict[m]['val']['targs'] =  val_targs
        model_dict[m]['val']['preds'] =  val_preds
        model_dict[m]['val']['probas'] =  val_probas
          
        model_dict[m]['test']['targs'] =  test_targs
        model_dict[m]['test']['preds'] =  test_preds
        model_dict[m]['test']['probas'] =  test_probas
        
        with open(data_path + "results\\results_" + m[0:-5] + ".pkl", "wb") as f:
            pickle.dump(model_dict, f)

    
    
         
