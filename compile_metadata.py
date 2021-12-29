import pandas as pd
import numpy as np

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
metadata_df.to_csv('data/processed/bill_metadata.csv')
print(f'Saving to data/processed/bill_metadata.csv')

