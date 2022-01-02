"""
This script fine-tunes a sentence bert model across several congressional bill documents.
The data consists of sentence pairs and a dummy variable indicating whether a sentence is in the same 
bill as the other sentence in the pair.  
"""

import pandas as pd
import random
import numpy as np
import logging
import math

import torch
from math import ceil, floor
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, LoggingHandler
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_data_and_rename_columns(path:str, rename_dict={}, drop_columns_dict={}, drop_congress_number=[])->pd.DataFrame:
    #load data
    df = pd.read_pickle(path)
    df = df.loc[~df['cong'].isin(drop_congress_number)].copy()
    df = df.rename(columns=rename_dict)
    df = df.drop(columns=drop_columns_dict)
    return df
    
def create_sentence_pairs(df:pd.DataFrame, number_of_sentence_pr_doc_max:int, random_seed:int)->list:
    sentence_pairs = [] #Output list of training examples
    random.seed(random_seed) #Set seed
    np.random.seed(random_seed) #Set seed

    #Create sentence pairs. Output is a list of dicts containing keys "texts" with two sentences and "label"
    for bill in tqdm(df['bill_id'].unique()):
        sentences_bill_x = df[df['bill_id']==bill]['sentences'].iat[0]
        n_sample = min(number_of_sentence_pr_doc_max, len(sentences_bill_x))
        sentence_one = random.sample(sentences_bill_x,k=n_sample) 
        same_document_dummy = np.random.choice([0, 1], size=(n_sample,))
        sentence_two_same_bill = [random.sample(sentences_bill_x,k=1) for _ in range(sum(same_document_dummy))]
        sentence_two_another_bill = [random.sample(df[df['bill_id']!=bill].sample(n=1,replace=False,random_state = random_seed)['sentences'].iat[0],k=1) for _ in range(n_sample - sum(same_document_dummy))]
        
        j = 0
        k = 0
        for x in range(len(sentence_one)):
            if same_document_dummy[x] == 1:
                sentence_pair = dict(texts=[sentence_one[x],sentence_two_same_bill[j][0]],label=1)
                j+=1
            elif same_document_dummy[x] == 0:
                sentence_pair = dict(texts=[sentence_one[x],sentence_two_another_bill[k][0]],label=0)
                k+=1
            sentence_pairs.append(sentence_pair)
    return sentence_pairs

def train_val_split(sentence_pairs, train_fraction, random_seed=3060):
    """Split data in training and validation"""
    train_obs_count = floor(len(sentence_pairs)*train_fraction)
    val_obs_count = ceil(len(sentence_pairs)*(1-train_fraction))
    train_data, val_data = torch.utils.data.random_split(sentence_pairs, [train_obs_count, val_obs_count],generator=torch.Generator().manual_seed(random_seed))
    return train_data, val_data

def prepare_training_data(training_data, batch_size = 16):
    training_data = [InputExample(texts=x['texts'],label=x['label']) for x in training_data.dataset]
    train_dataloader = DataLoader(training_data, shuffle=True, batch_size=batch_size)
    return train_dataloader

def prepare_validation_data(val_data:list):
    sentence1_val = [x['texts'][0] for x in val_data]
    sentence2_val = [x['texts'][1] for x in val_data]
    label_val = [x['label'] for x in val_data]
    return sentence1_val, sentence2_val, label_val

def main():
    logging.basicConfig(format='%(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                level=logging.INFO,
                handlers=[LoggingHandler()])

    logging.info("Loading data")
    df = load_data_and_rename_columns('data/processed/bert_data.pickle', drop_congress_number=[115])
    logging.info("Preparing sentence pairs...")
    #df = df.sample(n=300) #TODO: remove when training for real
    logging.info("Create sentence pairs")
    sentence_pairs = create_sentence_pairs(df, number_of_sentence_pr_doc_max=20, random_seed=3060)
    logging.info(f"{len(sentence_pairs)} sentence pairs are created")
    logging.info("Training-validation split...")
    train_data, val_data = train_val_split(sentence_pairs, train_fraction=0.9, random_seed=3060)
    logging.info(f"{len(train_data)} sentence pairs are in the training set")
    logging.info(f"{len(val_data)} sentence pairs are in the validation set")
    logging.info("Preparing training-data...")
    train_dataloader = prepare_training_data(train_data,batch_size = 32)
    logging.info("Preparing validation data...")
    sentence1_val, sentence2_val, label_val = prepare_validation_data(val_data)
    logging.info("Loading distilbert-base-nli-mean-tokens...")
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    num_epochs = 3
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
    evaluator = evaluation.BinaryClassificationEvaluator(sentence1_val,sentence2_val,label_val, name="validation metrics for the Binary Classifaction Evaluator", batch_size=32, show_progress_bar=True)
    train_loss = losses.SoftmaxLoss(model, model.get_sentence_embedding_dimension(), num_labels=2)
    model_save_path = "data/processed/pretrained_model"
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=num_epochs, warmup_steps=warmup_steps,evaluator=evaluator, evaluation_steps=len(train_dataloader), output_path=model_save_path)
    logging.info("Save model...")
    model.save(model_save_path)
    logging.info("Done")

if __name__ == '__main__':
    main()