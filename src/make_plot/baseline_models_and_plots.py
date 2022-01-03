import os

os.chdir("C:\\Users\\tnv664\\OneDrive - University of Copenhagen\\Documents\\Uni\\Ph.D\\deep learning\\deeplearningproject\\")
#os.chdir("C:\\Users\\augus\\OneDrive - Københavns Universitet\\Documents\\Uni\\Ph.D\\deep learning\\deeplearningproject")


from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import clear_output 

from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

import pickle
from random import choices
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

from src.make_plot.data_utils import straitified_train_validation_split, dataloader, metrics


tex = False #if true installs tex and the science style, which is not om colab - takes a littel while 
if tex:
  plt.style.use('science')
#else:
 # plt.style.use('science','no-latex')



path = 'data\\processed\\' 

model_path = "trained_models\\baseline\\" 

result_path = 'data\\results\\'

plot_path = 'plots_tables\\'

"""#Indlæser data"""

rerun_baseline = False #saves new prediction probas

if rerun_baseline:
  bert_train = torch.load(path + 'bert_train_103-114.pt')
  labels_train = torch.load(path + 'labels_train_103-114.pt')  
  
  test_dataset = torch.load(path + 'bert_test_115.pt')
  test_labels = torch.load(path + 'labels_test_115.pt')

  meta_train =  torch.load(path + 'meta_train_103-114.pt')
  meta_test = torch.load(path + 'meta_test_115.pt')

  bert_train, labels_train, bert_val, labels_val, indiciens_train, indicies_val = straitified_train_validation_split(bert_train, 
                                                                                   labels_train, 
                                                                                   idx = True)
  meta_train, meta_val = meta_train[indiciens_train], meta_train[indicies_val]

else:
  labels_train = torch.load(path + 'labels_train_103-114.pt')
  test_labels = torch.load(path + 'labels_test_115.pt')
  labels_train, labels_train, labels_val, labels_val = straitified_train_validation_split(labels_train, labels_train)

"""Naive baseline"""

share = sum(labels_train.numpy() == 1) / len(labels_train) 

print("The share in the train set is", share * 100)

population = [0, 1, ]
weights = [0.9654, 0.0346]

#random_val = [choices(population, weights)[0] for i in range(len(labels_val))] #random simulations
#random_test = [choices(population, weights)[0] for i in range(len(test_labels))]  #random simulations
random_val = [0] * len(labels_val) #predict the largest class
random_test =[0] * len(test_labels) #predict largest class

metrics(labels_val, random_val, test_labels, random_test)


run_log_only_meta = False
'''only meta logistic'''
if run_log_only_meta:
    clf_log_only_meta = LogisticRegression(random_state = 0, class_weight = 'balanced').fit(meta_train.numpy(), labels_train)
    val_preds = clf_log_only_meta.predict(meta_val.numpy())
    test_preds = clf_log_only_meta.predict(meta_test.numpy())
    
    filename = 'logreg_model_only_meta.sav'
    pickle.dump(clf_log_only_meta, open(model_path + filename, 'wb'))
    with open(result_path + 'logreg_only_meta_val_pred.npy', 'wb') as f:
        np.save(f,val_preds)
    with open(result_path + 'logreg_only_meta_test_pred.npy', 'wb') as f:
        np.save(f,test_preds)
else:
    filename = 'logreg_model_only_meta.sav'
    clf_log_only_meta = pickle.load(open(model_path + filename, 'rb'))
    with open(result_path + 'logreg_only_meta_val_pred.npy', 'rb') as f:
        val_preds = np.load(f)
    with open(result_path + 'logreg_only_meta_test_pred.npy', 'rb') as f:
        test_preds = np.load(f)

metrics(labels_val, val_preds, test_labels, test_preds)
auc_log_only_meta = roc_auc_score(labels_val, val_preds)
auc_log_only_meta_test = roc_auc_score(test_labels, test_preds)


if rerun_baseline:
  preds = clf_log_only_meta.predict_proba(meta_val.numpy())
  with open(result_path + 'logreg_only_meta_val_probas.npy', 'wb') as f:
    np.save(f,preds)
    
  preds_test = clf_log_only_meta.predict_proba(meta_test.numpy())
  with open(result_path + 'logreg_only_meta_test_probas.npy', 'wb') as f:
    np.save(f,preds_test)
else:
  with open(result_path + 'logreg_only_meta_val_probas.npy', 'rb') as f:
      preds = np.load(f)
  with open(result_path + 'logreg_only_meta_test_probas.npy', 'rb') as f:
      preds_test = np.load(f)
      
lr_precision_log_only_meta, lr_recall_log_only_meta , _ = precision_recall_curve(labels_val, preds[:,1])
lr_fpr_log_only_meta, lr_tpr_log_only_meta, _ = roc_curve(labels_val, preds[:,1])

lr_precision_log_only_meta_test, lr_recall_log_only_meta_test , _ = precision_recall_curve(test_labels, preds_test[:,1])
lr_fpr_log_only_meta_test, lr_tpr_log_only_meta_test, _ = roc_curve(test_labels, preds_test[:,1])


"""## Logistic regression"""

run_log = False #run logistic regression?

if run_log: #if we have not run the model or want to rerun it
  clf_log = LogisticRegression(random_state = 0, class_weight = 'balanced').fit(torch.squeeze(torch.mean(bert_train, axis = 2)).numpy(), labels_train)


  val_preds = clf_log.predict(torch.squeeze(torch.mean(bert_val, axis = 2)).numpy())
  test_preds = clf_log.predict(torch.squeeze(torch.mean(test_dataset, axis = 2)).numpy())

  filename = 'logreg_model.sav'
  pickle.dump(clf_log, open(model_path + filename, 'wb'))
  with open(result_path + 'logreg_val_pred.npy', 'wb') as f:
    np.save(f,val_preds)
  with open(result_path + 'logreg_test_pred.npy', 'wb') as f:
    np.save(f,test_preds)

else: #load in the model and the predicitons
  filename = 'logreg_model.sav'
  clf_log = pickle.load(open(model_path + filename, 'rb'))
  with open(result_path + 'logreg_val_pred.npy', 'rb') as f:
    val_preds = np.load(f)
  with open(result_path + 'logreg_test_pred.npy', 'rb') as f:
    test_preds = np.load(f)

metrics(labels_val, val_preds, test_labels, test_preds)

auc_log = roc_auc_score(labels_val, val_preds)
auc_log_test = roc_auc_score(test_labels, test_preds)


if rerun_baseline:
  preds = clf_log.predict_proba(torch.squeeze(torch.mean(bert_val, axis = 2)).numpy())
  with open(result_path + 'logreg_val_probas.npy', 'wb') as f:
    np.save(f,preds)
    
  preds_test = clf_log.predict_proba(torch.squeeze(torch.mean(test_dataset, axis = 2)).numpy())
  with open(result_path + 'logreg_test_probas.npy', 'wb') as f:
    np.save(f,preds_test) 
else:
  with open(result_path + 'logreg_val_probas.npy', 'rb') as f:
      preds = np.load(f)  
  with open(result_path + 'logreg_test_probas.npy', 'rb') as f:
      preds_test = np.load(f)  

lr_precision_log, lr_recall_log , _ = precision_recall_curve(labels_val, preds[:,1])
lr_fpr_log, lr_tpr_log, _ = roc_curve(labels_val, preds[:,1])

lr_precision_log_test, lr_recall_log_test , _ = precision_recall_curve(test_labels, preds_test[:,1])
lr_fpr_log_test, lr_tpr_log_test, _ = roc_curve(test_labels, preds_test[:,1])

"""## logistic regression with meta"""

run_log_meta = False

#with meta 
if run_log_meta: #if we have not run the model or want to rerun it
  clf_log_meta = LogisticRegression(random_state = 0, class_weight = 'balanced', max_iter = 1000).fit(np.append(torch.squeeze(torch.mean(bert_train, axis = 2)).numpy(), meta_train.numpy(), axis = 1),
                                                                                     labels_train)

  val_preds = clf_log_meta.predict(np.append(torch.squeeze(torch.mean(bert_val, axis = 2)).numpy(), meta_val.numpy(), axis = 1))
  test_preds = clf_log_meta.predict(np.append(torch.squeeze(torch.mean(test_dataset, axis = 2)).numpy(), meta_test.numpy(), axis = 1))
  filename = 'logreg_meta_model.sav'
  pickle.dump(clf_log_meta, open(model_path + filename, 'wb'))
  with open(result_path + 'logreg_meta_val_pred.npy', 'wb') as f:
    np.save(f,val_preds)
  with open(result_path + 'logreg_meta_test_pred.npy', 'wb') as f:
    np.save(f,test_preds)

else: #load in the model and the predicitons
  filename = 'logreg_meta_model.sav'
  clf_log_meta = pickle.load(open(model_path + filename, 'rb'))
  with open(result_path + 'logreg_meta_val_pred.npy', 'rb') as f:
    val_preds = np.load(f)
  with open(result_path + 'logreg_meta_test_pred.npy', 'rb') as f:
    test_preds =  np.load(f)

metrics(labels_val, val_preds, test_labels, test_preds)

auc_log_meta = roc_auc_score(labels_val, val_preds)
auc_log_meta_test = roc_auc_score(test_labels, test_preds)


if rerun_baseline:
  preds = clf_log_meta.predict_proba(np.append(torch.squeeze(torch.mean(bert_val, axis = 2)).numpy(), meta_val.numpy(), axis = 1))
  with open(result_path + 'logreg_val_meta_probas.npy', 'wb') as f:
   np.save(f,preds)
  preds_test = clf_log_meta.predict_proba(np.append(torch.squeeze(torch.mean(test_dataset, axis = 2)).numpy(), meta_test.numpy(), axis = 1))
  with open(result_path + 'logreg_test_meta_probas.npy', 'wb') as f:
   np.save(f,preds_test)   
   
else:
  with open(result_path + 'logreg_val_meta_probas.npy', 'rb') as f:
    preds = np.load(f)  
  with open(result_path + 'logreg_test_meta_probas.npy', 'rb') as f:
    preds_test = np.load(f)  

lr_precision_log_meta, lr_recall_log_meta , _ = precision_recall_curve(labels_val, preds[:,1])
lr_fpr_log_meta, lr_tpr_log_meta, _ = roc_curve(labels_val, preds[:,1])

lr_precision_log_meta_test, lr_recall_log_meta_test , _ = precision_recall_curve(test_labels, preds_test[:,1])
lr_fpr_log_meta_test, lr_tpr_log_meta_test, _ = roc_curve(test_labels, preds_test[:,1])

"""## Adaboost 

"""

run_ada = False #takes 1.5 hours to train again

if run_ada:

  clf_ada = AdaBoostClassifier(random_state = 0,  n_estimators = 5000, learning_rate = 1)
  clf_ada.fit(torch.squeeze(torch.mean(bert_train, axis = 2)).numpy(), labels_train)
  val_preds = clf_ada.predict(torch.squeeze(torch.mean(bert_val, axis = 2)).numpy())
  test_preds = clf_ada.predict(torch.squeeze(torch.mean(test_dataset, axis = 2)).numpy())


  filename = 'adaboost_model.sav'
  pickle.dump(clf_ada, open(model_path + filename, 'wb'))
  with open(result_path + 'ada_val_pred.npy', 'wb') as f:
    np.save(f,val_preds)
  with open(result_path + 'ada_test_pred.npy', 'wb') as f:
    np.save(f,test_preds)

else:
  filename = 'adaboost_model.sav'
  clf_ada = pickle.load(open(model_path + filename, 'rb'))

  with open(result_path + 'ada_val_pred.npy', 'rb') as f:
    val_preds = np.load(f)
  with open(result_path + 'ada_test_pred.npy', 'rb') as f:
    test_preds = np.load(f)

metrics(labels_val, val_preds, test_labels, test_preds)


auc_ada = roc_auc_score(labels_val, val_preds)
auc_ada_test = roc_auc_score(test_labels, test_preds)


if rerun_baseline:
  preds = clf_ada.predict_proba(torch.squeeze(torch.mean(bert_val, axis = 2)).numpy())
  with open(result_path + 'ada_val_probas.npy', 'wb') as f:
   np.save(f,preds)
  preds_test = clf_ada.predict_proba(torch.squeeze(torch.mean(test_dataset, axis = 2)).numpy())
  with open(result_path + 'ada_test_probas.npy', 'wb') as f:
   np.save(f,preds_test)
else:
  with open(result_path + 'ada_val_probas.npy', 'rb') as f:
    preds = np.load(f)  
  with open(result_path + 'ada_test_probas.npy', 'rb') as f:
    preds_test = np.load(f)  
    
lr_precision_ada, lr_recall_ada , _ = precision_recall_curve(labels_val, preds[:,1])
lr_fpr_ada, lr_tpr_ada, _ = roc_curve(labels_val, preds[:,1])

lr_precision_ada_test, lr_recall_ada_test , _ = precision_recall_curve(test_labels, preds_test[:,1])
lr_fpr_ada_test, lr_tpr_ada_test, _ = roc_curve(test_labels, preds_test[:,1])

"""## Adaboost with metadata"""

run_ada_meta = False #takes around 1.5 hours to train

if run_ada_meta:

  clf_ada_meta = AdaBoostClassifier(random_state = 0,  n_estimators = 5000, learning_rate = 1)
  clf_ada_meta.fit(np.append(torch.squeeze(torch.mean(bert_train, axis = 2)).numpy(), meta_train.numpy(), axis = 1), labels_train)
  val_preds = clf_ada_meta.predict(np.append(torch.squeeze(torch.mean(bert_val, axis = 2)).numpy(), meta_val.numpy(), axis = 1))
  test_preds = clf_ada_meta.predict(np.append(torch.squeeze(torch.mean(test_dataset, axis = 2)).numpy(), meta_test.numpy(), axis = 1))


  filename = 'adaboost_meta_model.sav'
  pickle.dump(clf_ada_meta, open(model_path + filename, 'wb'))
  with open(result_path + 'ada_meta_val_pred.npy', 'wb') as f:
    np.save(f,val_preds)
  with open(result_path + 'ada_meta_test_pred.npy', 'wb') as f:
    np.save(f,test_preds)

else:
  filename = 'adaboost_meta_model.sav'
  clf_ada_meta = pickle.load(open(model_path + filename, 'rb'))

  with open(result_path + 'ada_meta_val_pred.npy', 'rb') as f:
    val_preds = np.load(f)
  with open(result_path + 'ada_meta_test_pred.npy', 'rb') as f:
    test_preds = np.load(f)

metrics(labels_val, val_preds, test_labels, test_preds)

auc_ada_meta = roc_auc_score(labels_val, val_preds)
auc_ada_meta_test = roc_auc_score(test_labels, test_preds)


if rerun_baseline:        
  preds = clf_ada_meta.predict_proba(np.append(torch.squeeze(torch.mean(bert_val, axis = 2)).numpy(), meta_val.numpy(), axis = 1))
  with open(result_path + 'ada_meta_val_probas.npy', 'wb') as f:
   np.save(f,preds)
  preds_test = clf_ada_meta.predict_proba(np.append(torch.squeeze(torch.mean(test_dataset, axis = 2)).numpy(), meta_test.numpy(), axis = 1))
  with open(result_path + 'ada_meta_test_probas.npy', 'wb') as f:
   np.save(f,preds_test)
else:
  with open(result_path + 'ada_meta_val_probas.npy', 'rb') as f:
      preds = np.load(f)  
  with open(result_path + 'ada_meta_test_probas.npy', 'rb') as f:
      preds_test = np.load(f)  

lr_precision_ada_meta, lr_recall_ada_meta , _ = precision_recall_curve(labels_val, preds[:,1])
lr_fpr_ada_meta, lr_tpr_ada_meta, _ = roc_curve(labels_val, preds[:,1])


lr_precision_ada_meta_test, lr_recall_ada_meta_test , _ = precision_recall_curve(test_labels, preds_test[:,1])
lr_fpr_ada_meta_test, lr_tpr_ada_meta_test, _ = roc_curve(test_labels, preds_test[:,1])

"""Plots"""

####################################################Validation######################
ns_probs = [0.0346 for _ in range(len(labels_val))] #set to the random chance of 1
ns_fpr, ns_tpr, _ = roc_curve(labels_val, ns_probs)
auc_high = roc_auc_score(labels_val, ns_probs)
#with plt.style.context('science'):
fig, axs = plt.subplots(ncols = 2, dpi = 500, figsize=(8,3))
#with plt.style.context('science'):
axs[0].plot(lr_fpr_log, lr_tpr_log, '--+', markersize=2, label='Logistic regression - AUC: {}'.format(round(auc_log,2)), color = "red")
axs[0].plot(lr_fpr_ada, lr_tpr_ada, '--v', markersize=2, label='Adaboost - AUC: {}'.format(round(auc_ada,2)), color = "green")
axs[0].plot(ns_fpr, ns_tpr, linestyle='-.', label='Random Classifier - AUC: {}'.format(round(auc_high,2)), color = "blue")
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC')
axs[0].legend(fontsize=6)

axs[1].plot(lr_precision_log, lr_recall_log, '--+', markersize=2, label='Logistic regression', color = "red")
axs[1].plot(lr_precision_ada, lr_recall_ada, '--v', markersize=2, label='Adaboost', color = "green")

random = no_skill = len([t for t in labels_val if t ==1]) / len(labels_val)
#pr_rand, recall_rand, _ = precision_recall_curve(labels_val, ns_probs)
axs[1].plot([0,1], [random,random], linestyle='-.', label='Random Classifier', color = "blue")
axs[1].set_ylabel('Recall')
axs[1].set_xlabel('Precission')
axs[1].set_title('Precision-Recall')
axs[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(plot_path + 'baseline_metrics.pdf', format='pdf')

ns_probs = [0.0346 for _ in range(len(labels_val))] #set to the random chance of 1
ns_fpr, ns_tpr, _ = roc_curve(labels_val, ns_probs)
auc_high = roc_auc_score(labels_val, ns_probs)
#with plt.style.context('science'):
fig, axs = plt.subplots(ncols = 2, dpi = 500, figsize=(8,3))
#with plt.style.context('science'):
axs[0].plot(lr_fpr_log_only_meta, lr_tpr_log_only_meta, '--+', markersize=2, label='Logistic regression (only meta) - AUC: {}'.format(round(auc_log_only_meta,2)), color = "black")
axs[0].plot(lr_fpr_log_meta, lr_tpr_log_meta, '--+', markersize=2, label='Logistic regression - AUC: {}'.format(round(auc_log_meta,2)), color = "red")
axs[0].plot(lr_fpr_ada_meta, lr_tpr_ada_meta, '--v', markersize=2, label='Adaboost - AUC: {}'.format(round(auc_ada_meta,2)), color = "green")
axs[0].plot(ns_fpr, ns_tpr, linestyle='-.', label='Random Classifier - AUC: {}'.format(round(auc_high,2)), color = "blue")
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC with meta data')
axs[0].legend(fontsize=6)

axs[1].plot(lr_precision_log_only_meta, lr_recall_log_only_meta, '--+', markersize=2, label='Logistic regression (only meta)', color = "black")
axs[1].plot(lr_precision_log_meta, lr_recall_log_meta, '--+', markersize=2, label='Logistic regression', color = "red")
axs[1].plot(lr_precision_ada_meta, lr_recall_ada_meta, '--v', markersize=2, label='Adaboost', color = "green")

random = no_skill = len([t for t in labels_val if t ==1]) / len(labels_val)
#pr_rand, recall_rand, _ = precision_recall_curve(labels_val, ns_probs)
axs[1].plot([0,1], [random,random], linestyle='-.', label='Random Classifier', color = "blue")
axs[1].set_ylabel('Recall')
axs[1].set_xlabel('Precission')
axs[1].set_title('Precision-Recall with meta data')
axs[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(plot_path + 'baseline_meta_metrics.pdf', format='pdf')


################################################################# Test

ns_probs = [0.0346 for _ in range(len(test_labels))] #set to the random chance of 1
ns_fpr, ns_tpr, _ = roc_curve(test_labels, ns_probs)
auc_high = roc_auc_score(test_labels, ns_probs)
#with plt.style.context('science'):
fig, axs = plt.subplots(ncols = 2, dpi = 500, figsize=(8,3))
#with plt.style.context('science'):
axs[0].plot(lr_fpr_log_test, lr_tpr_log_test, '--+', markersize=2, label='Logistic regression (only meta) - AUC: {}'.format(round(auc_log_only_meta_test,2)), color = "black")
axs[0].plot(lr_fpr_ada_test, lr_tpr_ada_test, '--v', markersize=2, label='Adaboost - AUC: {}'.format(round(auc_ada_test,2)), color = "green")
axs[0].plot(ns_fpr, ns_tpr, linestyle='-.', label='Random Classifier - AUC: {}'.format(round(auc_high,2)), color = "blue")
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC')
axs[0].legend(fontsize=6)

axs[1].plot(lr_precision_log_test, lr_recall_log_test, '--+', markersize=2, label='Logistic regression', color = "red")
axs[1].plot(lr_precision_ada_test, lr_recall_ada_test, '--v', markersize=2, label='Adaboost', color = "green")

random = no_skill = len([t for t in labels_val if t ==1]) / len(labels_val)
#pr_rand, recall_rand, _ = precision_recall_curve(labels_val, ns_probs)
axs[1].plot([0,1], [random,random], linestyle='-.', label='Random Classifier', color = "blue")
axs[1].set_ylabel('Recall')
axs[1].set_xlabel('Precission')
axs[1].set_title('Precision-Recall')
axs[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(plot_path + 'baseline_test_metrics.pdf', format='pdf')

#with plt.style.context('science'):
fig, axs = plt.subplots(ncols = 2, dpi = 500, figsize=(8,3))
#with plt.style.context('science'):
axs[0].plot(lr_fpr_log_only_meta_test, lr_tpr_log_only_meta_test, '--+', markersize=2, label='Logistic regression (only meta) - AUC: {}'.format(round(auc_log_only_meta_test,2)), color = "black")
axs[0].plot(lr_fpr_log_meta_test, lr_tpr_log_meta_test, '--+', markersize=2, label='Logistic regression - AUC: {}'.format(round(auc_log_meta_test,2)), color = "red")
axs[0].plot(lr_fpr_ada_meta_test, lr_tpr_ada_meta_test, '--v', markersize=2, label='Adaboost - AUC: {}'.format(round(auc_ada_meta_test,2)), color = "green")
axs[0].plot(ns_fpr, ns_tpr, linestyle='-.', label='Random Classifier - AUC: {}'.format(round(auc_high,2)), color = "blue")
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC with meta data')
axs[0].legend(fontsize=6)

axs[1].plot(lr_precision_log_only_meta_test, lr_recall_log_only_meta_test, '--+', markersize=2, label='Logistic regression', color = "black")
axs[1].plot(lr_precision_log_meta_test, lr_recall_log_meta_test, '--+', markersize=2, label='Logistic regression', color = "red")
axs[1].plot(lr_precision_ada_meta_test, lr_recall_ada_meta_test, '--v', markersize=2, label='Adaboost', color = "green")

random = no_skill = len([t for t in labels_val if t ==1]) / len(labels_val)
#pr_rand, recall_rand, _ = precision_recall_curve(labels_val, ns_probs)
axs[1].plot([0,1], [random,random], linestyle='-.', label='Random Classifier', color = "blue")
axs[1].set_ylabel('Recall')
axs[1].set_xlabel('Precission')
axs[1].set_title('Precision-Recall with meta data')
axs[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(plot_path + 'baseline_test_meta_metrics.pdf', format='pdf')



######################################################################loss#################
"""import losses"""

df = pd.read_pickle(result_path + "train_val_metrics_2021-12-31.pkl")
keys = df.keys()


figure(figsize=(4, 3), dpi= 500)

for key in keys:
  plt.plot(range(len(df[key]['train_loss'])),df[key]['train_loss'], label = key + " loss : {}".format(round(df[key]['train_loss'].values[-1],2)))

plt.title("Train Loss over epochs")
plt.legend(fontsize=6)

plt.show()
plt.tight_layout()
plt.savefig(plot_path + 'train_losses.pdf', format='pdf')

######################################################## import predictions and do metrics and plots



with open(result_path + "predictions.pkl", "rb") as f:
    results = pickle.load(f)

##
#plot_dict = {}
#networks_path = "src\\models\\trained_models\\"

#for m in os.listdir(model_path):
for m in results.keys():
            
    #metrics
    print(m)
    metrics(results[m]['val']['targs'],
            results[m]['val']['preds'],
            results[m]['test']['targs'],
            results[m]['test']['preds'])
    
####################################Validation################################   
####### non meta plot 

markers = ['-+','-*','-^']
#with plt.style.context('science'):
fig, axs = plt.subplots(ncols = 2, dpi = 500, figsize=(8,3))
#with plt.style.context('science'):
axs[0].plot(lr_fpr_log, lr_tpr_log, '--+', markersize=2, label='Logistic regression - AUC: {}'.format(round(auc_log,2)), color = "red")
axs[1].plot(lr_precision_log, lr_recall_log, '--+', markersize=2, label='Logistic regression', color = "red")

i = 0
for m in results.keys():
    if 'meta' not in m:
        auc = roc_auc_score(results[m]['val']['targs'],results[m]['val']['preds'])
        axs[0].plot(results[m]['val']['fpr'], results[m]['val']['tpr'], markers[i], markersize=3, label= m + '- AUC: {}'.format(round(auc,2)))
        axs[1].plot(results[m]['val']['pr'], results[m]['val']['recal'], markers[i], markersize=3, label= m)
        i+= 1
    
    #axs[1].plot(lr_precision_ada, lr_recall_ada, '--v', markersize=2, label='Adaboost', color = "green")
#axs[0].plot(lr_fpr_ada, lr_tpr_ada, '--v', markersize=2, label='Adaboost - AUC: {}'.format(round(auc_ada,2)), color = "green")
#axs[0].plot(ns_fpr, ns_tpr, linestyle='-.', label='Random Classifier - AUC: {}'.format(round(auc_high,2)), color = "blue")
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC')
axs[0].legend(fontsize=6)
axs[1].set_ylabel('Recall')
axs[1].set_xlabel('Precission')
axs[1].set_title('Precision-Recall')
axs[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(plot_path + 'trained_metrics.pdf', format='pdf')



fig, axs = plt.subplots(ncols = 2, dpi = 500, figsize=(8,3))
#with plt.style.context('science'):
axs[0].plot(lr_fpr_log_meta, lr_tpr_log_meta, '--+', markersize=2, label='Logistic regression - AUC: {}'.format(round(auc_log,2)), color = "red")
axs[1].plot(lr_precision_log_meta, lr_recall_log_meta, '--+', markersize=2, label='Logistic regression', color = "red")

i = 0
for m in results.keys():
    if 'meta' in m:
        auc = roc_auc_score(results[m]['val']['targs'],results[m]['val']['preds'])
        axs[0].plot(results[m]['val']['fpr'], results[m]['val']['tpr'], markers[i], markersize=3, label= m + '- AUC: {}'.format(round(auc,2)))
        axs[1].plot(results[m]['val']['pr'], results[m]['val']['recal'], markers[i], markersize=3, label= m)    
        i +=1
    #axs[1].plot(lr_precision_ada, lr_recall_ada, '--v', markersize=2, label='Adaboost', color = "green")
#axs[0].plot(lr_fpr_ada, lr_tpr_ada, '--v', markersize=2, label='Adaboost - AUC: {}'.format(round(auc_ada,2)), color = "green")
#axs[0].plot(ns_fpr, ns_tpr, linestyle='-.', label='Random Classifier - AUC: {}'.format(round(auc_high,2)), color = "blue")
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC')
axs[0].legend(fontsize=6)
axs[1].set_ylabel('Recall')
axs[1].set_xlabel('Precission')
axs[1].set_title('Precision-Recall')
axs[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(plot_path + 'trained_meta_metrics.pdf', format='pdf')



######################################### test ###################################

markers = ['-+','-*','-^']
#with plt.style.context('science'):
fig, axs = plt.subplots(ncols = 2, dpi = 500, figsize=(8,3))
#with plt.style.context('science'):
axs[0].plot(lr_fpr_log_test, lr_tpr_log_test, '--+', markersize=2, label='Logistic regression - AUC: {}'.format(round(auc_log_test,2)), color = "red")
axs[1].plot(lr_precision_log_test, lr_recall_log_test, '--+', markersize=2, label='Logistic regression', color = "red")

i = 0
for m in results.keys():
    if 'meta' not in m:
        auc = roc_auc_score(results[m]['test']['targs'],results[m]['test']['preds'])
        axs[0].plot(results[m]['test']['fpr'], results[m]['test']['tpr'], markers[i], markersize=3, label= m + '- AUC: {}'.format(round(auc,2)))
        axs[1].plot(results[m]['test']['pr'], results[m]['test']['recal'], markers[i], markersize=3, label= m)
        i+= 1
    
    #axs[1].plot(lr_precision_ada, lr_recall_ada, '--v', markersize=2, label='Adaboost', color = "green")
#axs[0].plot(lr_fpr_ada, lr_tpr_ada, '--v', markersize=2, label='Adaboost - AUC: {}'.format(round(auc_ada,2)), color = "green")
#axs[0].plot(ns_fpr, ns_tpr, linestyle='-.', label='Random Classifier - AUC: {}'.format(round(auc_high,2)), color = "blue")
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC')
axs[0].legend(fontsize=6)
axs[1].set_ylabel('Recall')
axs[1].set_xlabel('Precission')
axs[1].set_title('Precision-Recall')
axs[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(plot_path + 'trained_test_metrics.pdf', format='pdf')



fig, axs = plt.subplots(ncols = 2, dpi = 500, figsize=(8,3))
#with plt.style.context('science'):
axs[0].plot(lr_fpr_log_meta_test, lr_tpr_log_meta_test, '--+', markersize=2, label='Logistic regression - AUC: {}'.format(round(auc_log_meta_test,2)), color = "red")
axs[1].plot(lr_precision_log_meta_test, lr_recall_log_meta_test, '--+', markersize=2, label='Logistic regression', color = "red")

i = 0
for m in results.keys():
    if 'meta' in m:
        auc = roc_auc_score(results[m]['test']['targs'],results[m]['test']['preds'])
        axs[0].plot(results[m]['test']['fpr'], results[m]['test']['tpr'], markers[i], markersize=3, label= m + '- AUC: {}'.format(round(auc,2)))
        axs[1].plot(results[m]['test']['pr'], results[m]['test']['recal'], markers[i], markersize=3, label= m)    
        i +=1
    #axs[1].plot(lr_precision_ada, lr_recall_ada, '--v', markersize=2, label='Adaboost', color = "green")
#axs[0].plot(lr_fpr_ada, lr_tpr_ada, '--v', markersize=2, label='Adaboost - AUC: {}'.format(round(auc_ada,2)), color = "green")
#axs[0].plot(ns_fpr, ns_tpr, linestyle='-.', label='Random Classifier - AUC: {}'.format(round(auc_high,2)), color = "blue")
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('ROC')
axs[0].legend(fontsize=6)
axs[1].set_ylabel('Recall')
axs[1].set_xlabel('Precission')
axs[1].set_title('Precision-Recall')
axs[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(plot_path + 'trained_meta_test_metrics.pdf', format='pdf')




######################################### Comparison between validation and test##########################