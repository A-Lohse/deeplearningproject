import os

#need to set the dir the the base of the project 
#os.chdir("C:\\Users\\tnv664\\OneDrive - University of Copenhagen\\Documents\\Uni\\Ph.D\\deep learning\\deeplearningproject")
os.chdir("C:\\Users\\augus\\OneDrive - Københavns Universitet\\Documents\\Uni\\Ph.D\\deep learning\\deeplearningproject\\src")
from make_plot.data_utils import  metrics

from models.sbert_downstream_CNN import SBertDsCNN
from models.sbert_downstream_FNN import SBertDsFNN
from data_modules import sbert_downstream_datamodule
import torch
import pickle
from sklearn.metrics import precision_recall_curve, roc_curve

model_path = "C:\\Users\\augus\\OneDrive - Københavns Universitet\\Documents\\Uni\\Ph.D\\deep learning\\deeplearningproject\\trained_models\\"
data_path = "C:\\Users\\augus\\OneDrive - Københavns Universitet\\Documents\\Uni\\Ph.D\\deep learning\\deeplearningproject\\data\\"

model_dict = {}

for m in os.listdir(model_path):
    print(m)
   #if 'meta' in m: #for now
    #    continue
    if "baseline" in m: #do not do anything to the baseline models 
        continue
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
        dm = sbert_downstream_datamodule.SbertDSDataModule(data_path = data_path + "processed")
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
        
                if 'meta' in m:        
                    outputs = net(local_batch.float(), local_meta.float())
                else:
                    outputs = net(local_batch.float())
                    
                predicted = torch.max(outputs.data, 1)[1]
                val_probas += list(outputs.data.numpy()[:,1])
                val_targs += list(local_labels.numpy())
                val_preds += list(predicted.data.numpy())
        model_dict[m]['val']['targs'] =  val_targs
        model_dict[m]['val']['preds'] =  val_preds
        model_dict[m]['val']['probas'] =  val_probas
        
        #clear up memory 
        del val_probas, val_targs, val_preds, outputs, predicted, local_batch, local_meta, local_labels
            ####TEST DATA####
        with torch.no_grad():
            for local_batch, local_meta, local_labels in dm.test_dataloader():
                if "FNN" in m:
                    if 'avg' in m:
                        local_batch = torch.mean(local_batch,axis = 2) #take the mean of the sentences in each document (dim 2 (with 0 indexing))
                        local_batch = torch.squeeze(local_batch) #remove the channel dim we put there for CNN
                    else:
                        local_batch.flatten(start_dim = 1) #concat the the input
                
                if 'meta' in m:        
                    outputs = net(local_batch.float(), local_meta.float())
                else:
                    outputs = net(local_batch.float())
                    
                predicted = torch.max(outputs.data, 1)[1]
                test_probas += list(outputs.data.numpy()[:,1])
                test_targs += list(local_labels.numpy())
                test_preds += list(predicted.data.numpy())
        
        

          
        model_dict[m]['test']['targs'] =  test_targs
        model_dict[m]['test']['preds'] =  test_preds
        model_dict[m]['test']['probas'] =  test_probas
        
        del net, dm, my_model, test_probas, test_targs, test_preds, outputs, predicted, local_batch, local_meta, local_labels



#create some nice metrics for plotting 
results = model_dict
for m in results.keys():
            
    #metrics
    metrics(results[m]['val']['targs'],
            results[m]['val']['preds'],
            results[m]['test']['targs'],
            results[m]['test']['preds'])
    
    pr_val, recal_val , _ = precision_recall_curve(results[m]['val']['targs'], results[m]['val']['probas'])
    pr_test, recal_test , _ = precision_recall_curve(results[m]['test']['targs'], results[m]['test']['probas'])
    
    #asign values
    
    
    fpr_val, tpr_val, _ = roc_curve(results[m]['val']['targs'], results[m]['val']['probas'])
    fpr_test, tpr_test, _ = roc_curve(results[m]['test']['targs'], results[m]['test']['probas'])
    
    #asign the values
    results[m]['val']['pr'] = pr_val
    results[m]['val']['recal'] = recal_val
    results[m]['val']['fpr'] = fpr_val
    results[m]['val']['tpr'] = tpr_val
    
    results[m]['test']['pr'] = pr_test
    results[m]['test']['recal'] = recal_test
    results[m]['test']['fpr'] = fpr_test
    results[m]['test']['tpr'] = tpr_test        


with open(data_path + "results\\predictions.pkl", "wb") as f:
    pickle.dump(model_dict, f)

    
