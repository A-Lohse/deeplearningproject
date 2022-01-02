"""
This script is used to prepare the bill data both for sBERT finetuning
and extracting sBERT embeddings. The data can later be split into train
and test based on congress number. The final output is a dataframe with 
the following columns:
--------------------------------------------------------------------------------
-    bill_id: the id of the bill e.g. "103_hr1002"
-    sentences: list of sentences in the bill to be used for BERT finetuning
-    cong: congress number
-    bill_status: If the bill was passed or not
--------------------------------------------------------------------------------
The original data comes from  "BillSum: A Corpus for Automatic Summarization of US Legislation"
project. See the original code repository at:  https://github.com/FiscalNote/BillSum
and the specific required files "us_train_sent_scores.pk" and "us_test_sent_scores.pkl"
"""
import pandas as pd
import pickle

#Read in bill meta data including bill status
bills_meta_path = 'data/processed/bills_meta.csv'

#Read in bill text from Billsum project
raw_train_path = 'data/raw/us_train_sent_scores.pkl'
raw_test_path = 'data/raw/us_test_sent_scores.pkl'
#BERT save path. 
processed_path = 'data/processed/bert_data.pickle'

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
    #Subset on "real text" and group sentences by BILL id.
    df = pd.DataFrame(bills, columns=['sentences', 'real_text', 'bill_id'])
   
    return pd.DataFrame(df.loc[df['real_text']==1]\
            .groupby('bill_id')['sentences'].apply(list))

def main():
    print(f'Reading bill text from Billsum data located at {raw_train_path} and {raw_test_path}')
    train = read_bills(raw_train_path)
    test = read_bills(raw_test_path)
    print(f'Concatinating training and test data')
    bert_df = pd.concat([train, test]).reset_index()
    og_obs = bert_df.shape[0]
    print(f'Billsum data shape: {bert_df.shape}')
    #Read in bill meta data including bill status
    print(f'Reading bill meta data from {bills_meta_path}')
    bills_meta = pd.read_csv(bills_meta_path, index_col = 0)
    print(f'There are  {bills_meta.shape[0]} bills in the meta dataset')
    #Merge the bill meta data with the bill text
    print(f'Doing an inner join of bill meta data with bill text...')
    bert_df = bert_df.merge(bills_meta, on='bill_id', how='inner')
    print(f'Final dataset has {bert_df.shape[0]} bills\n missing {og_obs - bills_meta.shape[0]} bills')
    print(f'Printing first 5 rows of the dataset')
    print(bert_df.head())
    #Save final data
    print(f'Saving dataset to {processed_path}')
    bert_df.to_pickle(processed_path)
    print('Done!')

if __name__ == '__main__':
    main()