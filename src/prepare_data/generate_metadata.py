"""
This scripts is used to process the metadata feed to our models
data comes from the congressional Bill project: http://congressionalbills.org/
"""

import pandas as pd
import numpy as np
import torch

df114 = pd.read_csv('data/raw/bills93-114.csv', sep=';', encoding='latin-1')
df115 = pd.read_csv('data/raw/bills115-116.csv')
metadata_df = pd.concat([df114, df115])

bill_status_df = pd.read_csv('data/raw/bill_status.csv', index_col=0) 

def harmonize_bill_id(bill_id):
    """
    Updates bill_id to harmonize with bill_status.csv
    """
    bill_id = bill_id.lower()
    bill_id = bill_id.replace('-', '_')
    bill_id = bill_id.split('_')
    try:
        bill_id = bill_id[0]+'_'+bill_id[1]+bill_id[2]
    except IndexError:
        bill_id = np.nan
    return bill_id

metadata_df['BillID'] = metadata_df['BillID'].apply(harmonize_bill_id)
metadata_df.rename(columns={'BillID':'bill_id'}, inplace=True)

all_ids = bill_status_df['bill_id'].tolist()
metadata_df = bill_status_df.merge(metadata_df, on='bill_id', how='left')
metadata_df.drop_duplicates(subset=['bill_id'], inplace=True)
newids = metadata_df['bill_id'].tolist()
print(f'Missing set of ids: {set(all_ids)-set(newids)}')

#select explanatory variables and get dummies
keepcols = ['bill_id', 'status', 'cong', 'Cosponsr', 'Majority', 'Party', 'Gender']
metadata_df = metadata_df[keepcols]
metadata_df[['republican', 'independent']] = pd.get_dummies(metadata_df['Party'], drop_first=True)
metadata_df.drop(columns=['Party'], inplace=True)

metadata_df = metadata_df.rename(columns={'Cosponsr':'cosponsors', 'Majority':'majority', 'Gender':'gender'})
#save metadata
metadata_df.to_csv('data/processed/bills_metadata.csv', index=False)

#override keepcols for final
keepcols = ['cosponsors', 'majority', 'republican', 'independent', 'cong', 'gender']
metadata_df = metadata_df[keepcols]
train_meta = metadata_df.loc[metadata_df['cong'] != 115].drop(columns=['cong'])
test_meta = metadata_df.loc[metadata_df['cong'] == 115].drop(columns=['cong'])

#convert to tensors
train_meta = torch.tensor(train_meta.to_numpy()).nan_to_num()
test_meta = torch.tensor(test_meta.to_numpy()).nan_to_num()

#save tensors
torch.save(train_meta, 'data/processed/meta_train_103-114.pt')
torch.save(test_meta, 'data/processed/meta_test_115.pt')


