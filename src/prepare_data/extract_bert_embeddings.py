"""
This script is used to extract the sBERT embeddings.

"""

#imports
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
import torch
from sentence_transformers import SentenceTransformer

# if fintuned model is used
fine_tuned_sbert = True

if fine_tuned_sbert:
    sBERT = SentenceTransformer('trained_models/finetuned_sbert')
    prefix = 'finetuned_sbert_'
else:
    sBERT = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    prefix = 'sbert_'

#verify if CUDA is available
print(f'Running CUDA on GPU: {torch.cuda.is_available()}')
#set path bert data
sbert_data_path = 'data/processed/bert_data.pickle'
#set path to dump processed data
processed_path = 'data/processed/'
#Load the SentenceTransformer model
print('Loading Sentence Transformer model...')
#Load in pre-trained sBERT model
#Or read in fine-tuned sBERT model
#sBERT = SentenceTransformer('changepathhere')
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
    
    emb = np.zeros((ndocs, maxsents, 384))
    for i, doc in enumerate(tqdm(docs)):
        for j, sent in enumerate(doc):
            emb[i,j,:] = model.encode(sent)
    emb = torch.tensor(emb, dtype=torch.float)
    return emb[:,None,:,:]

def main():
    print('Reading sBERT data...')
    bert_data = pd.read_pickle(sbert_data_path)
    print(f"Saving training and test indices to {processed_path + 'train_test_indices.pickle'}")
    train_test_idx = {'train_idx':bert_data.loc[bert_data['cong'] != 115].index.tolist(),
                      'test_idx':bert_data.loc[bert_data['cong'] == 115].index.tolist()}
    
    #Extract the bill documents
    bill_docs = bert_data['sentences'].tolist()
    #Extract the number of bills and max number of sentences 
    nbills = bert_data.shape[0]
    max_sents = bert_data['sentences'].apply(lambda x: len(x)).max()
    #Create the feature matrix
    print(f'Creating the embeddings matrix of shape:\n\
           - nbills: {nbills}\n\
           - max_sents: {max_sents}\n\
           - emb_size: 384')
    embeddings = docsents_embeddings(sBERT, bill_docs, nbills, max_sents)
    labels = torch.tensor(bert_data['status'], dtype=torch.long)
    print(f'Finished creating embeddings matrix')
    
    train_embs = embeddings[train_test_idx['train_idx']]
    test_embs = embeddings[train_test_idx['test_idx']]
    train_labels = labels[train_test_idx['train_idx']]
    test_labels = labels[train_test_idx['test_idx']]
    print(f'Saving training and test embeddings and labels to {processed_path}')
    torch.save(train_embs, processed_path + prefix + 'train_103-114.pt')
    torch.save(test_embs, processed_path +  prefix + 'test_115.pt')
    try:
        torch.load(processed_path + prefix + 'labels_train_103-114.pt')
        torch.load(test_labels, processed_path + prefix + 'labels_test_115.pt')
    except:
        torch.save(train_labels, processed_path + prefix + 'labels_train_103-114.pt')
        torch.save(test_labels, processed_path + prefix + 'labels_test_115.pt')
  
    print('FINISHED!')

if __name__ == '__main__':
    main()

