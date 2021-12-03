"""
This script is used to prepare the data from the paper 
"BillSum: A Corpus for Automatic Summarization of US Legislation"
See the original code repository at: https://github.com/FiscalNote/BillSum
and the specific required files "us_train_sent_scores.pk" and "us_test_sent_scores.pkl"
can be downloaded from: https://drive.google.com/file/d/1uBCRSs_KFv7jD6nM4MKXZZ4nZAPI2Go4/view
"""

#imports
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
import torch
from sentence_transformers import SentenceTransformer

#verify if CUDA is available
print(f'Running CUDA on GPU: {torch.cuda.is_available()}')
#set path to raw data
raw_train_path = 'data/raw/us_train_sent_scores.pkl'
raw_test_path = 'data/raw/us_test_sent_scores.pkl'
#set path to dump processed data
processed_path = 'data/processed/'
#Load the SentenceTransformer model
print('Loading Sentence Transformer model...')
sBERT = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def read_bills(path:str):
    """
    This function is used to get the bill id from the file name.
    input:
        path(str): path to the file
    returns:
        data(pandas.DataFrame): DataFrame with bill id and sentences
    """
    sent_data = pickle.load(open(path, 'rb'))
    doc_order = sorted(sent_data.keys())
    bills = []
    for key in doc_order:
        doc = sent_data[key]
        for sent in doc:
            y = int(sent[2]['rouge-2']['p'] > 0.1)
            # Our label 
            bills.append([sent[0], y, key])
    return pd.DataFrame(bills, columns=['sentence', 'real_text', 'bill_id'])


def docsents_embeddings(model:SentenceTransformer, docs:List[list],
                        ndocs:int, maxsents:int)->np.array:
    """
    Creates the feature matrix of sBERT document embeddings
    for downstream classification.
    Input:
        model(SentenceTransformer): SentenceTransformer object
        docs(list[list]): List of documents. Each document is a list of sentences.
        ndocs(int):Number of documents
        maxsents(int):Max amount of sentences
    Returns:
        N_docs x Max_sentences x 384 dimensional np.array with BERT embeddings
        for each document.
    """
    DM = np.zeros((ndocs, maxsents, 384))
    for i, doc in enumerate(tqdm(docs)):
        for j, sent in enumerate(doc):
            embedding = model.encode(sent)
            DM[i,j,:] = embedding
    return DM

def main():
    print('Reading raw data...')
    train_dat = read_bills(raw_train_path)
    test_dat = read_bills(raw_test_path)
    #Subset on "real text" and group sentences by BILL id.
    train_dat = pd.DataFrame(train_dat.loc[train_dat['real_text']==1]\
        .groupby('bill_id')['sentence'].apply(list))
    test_dat = pd.DataFrame(test_dat.loc[test_dat['real_text']==1]\
        .groupby('bill_id')['sentence'].apply(list))
    ndocs_train = len(train_dat)
    ndocs_test = len(test_dat)
    print(f'There are {ndocs_train} bills in training set')
    print(f'There are {ndocs_test} bills in training set')
    train_max_sents = train_dat['sentence'].apply(lambda x: len(x)).max()
    test_max_sents = test_dat['sentence'].apply(lambda x: len(x)).max()
    print(f'Max number of sentences in one bill from train: {train_max_sents}')
    print(f'Max number of sentences in one bill from test: {test_max_sents}')
    #Saving the processed text data
    print(f'Saving processed data to {processed_path}')
    train_dat.to_pickle(processed_path + 'train_dat.pkl')
    test_dat.to_pickle(processed_path + 'test_dat.pkl')
    #Save the training and test labels
    train_labels = train_dat.index.tolist()
    test_labels = test_dat.index.tolist()
    #Saving the training and test labels
    print(f'Saving the training and test labels to {processed_path}')
    trainslabs_fname = processed_path + 'train_labels.pkl'
    testslabs_fname = processed_path + 'test_labels.pkl'
    print(f'Saving the training labels to {trainslabs_fname}')
    print(f'Saving the test labels to {testslabs_fname}')
    pickle.dump(train_labels, open(trainslabs_fname, 'wb'))
    pickle.dump(test_labels, open(testslabs_fname, 'wb'))
    #Extract sBERT embeddings for training and test data
    print('Extracting sBERT embeddings for training...')
    train_docs = train_dat['sentence'].tolist()
    train_embs = docsents_embeddings(sBERT, train_docs, 
                                     ndocs_train, train_max_sents)
    train_fname = processed_path + 'train_embs.npy'
    print(f'Saving training embeddings to {train_fname}')
    with open(train_fname, 'wb') as f:
        np.save(f, train_embs)
    print('Extracting sBERT embeddings for test...')
    test_docs = test_dat['sentence'].tolist()
    #Use the same max sentences for training and test 
    #to have matching dimensions
    test_embs = docsents_embeddings(sBERT, test_docs, 
                                    ndocs_test, train_max_sents)
    test_fname = processed_path + 'test_embs.npy'
    print(f'Saving test embeddings to {test_fname}')
    with open(test_fname, 'wb') as f:
        np.save(f, test_embs)
    print('FINISHED!')

if __name__ == '__main__':
    main()

