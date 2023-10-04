import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
import datetime
import sqlite3
import sys
from pandas import read_sql_query, read_sql_table
from tqdm import tqdm

sys.path.append('/mnt/zfsusers/hollowayp/kSWAP/kswap/')

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)

#file_path = './data/'+sys.argv[1]+'.db'
from config import Config
file_path = Config().examples_path+'/data/'+Config().db_name
file_path = '/mnt/zfsusers/hollowayp/kSWAP/examples/data//swap_beta_test_db_Sept2023_beta_FINAL_from_csv.db'
print('filepath',file_path)
print(sys.argv)
def thresholds_setting():
    from config import Config as config_0
    sys.path.insert(0, '../kswap')
    p_real = config_0().thresholds[1]
    p_bogus = config_0().thresholds[0]
    return [p_real,p_bogus]

def prior_setting():
    from config import Config as config_0
    return config_0().p0

def read_sqlite(dbfile):
    with sqlite3.connect(dbfile) as dbcon:
      tables = list(read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
      out = {tbl : read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}
    return out

print('USERS')
print('User db')
user_db =pd.DataFrame.from_dict(read_sqlite(file_path)['users']) 
user_db.to_csv(Config().examples_path+'/data/DES_Beta_Sept_2023_users.csv')

print(user_db[['user_id', 'user_score', 'confusion_matrix']])

print('User history')
#print(pd.DataFrame.from_dict(read_sqlite(file_path)['users'])['history'][0])
print('SUBJECTS')
subj_db = pd.DataFrame.from_dict(read_sqlite(file_path)['subjects'])
subj_db['score']= subj_db['score'].astype('float64')
print('Sum not retired:',len(subj_db[(subj_db['score']>=1e-5)&(subj_db['gold_label']==-1)]))
print('Sum test',len(set(subj_db['subject_id'])))
subj_db.to_csv(Config().examples_path+'/data/DES_Beta_Sept_2023_subjects.csv')

print('Test subjects')
print(subj_db[subj_db['gold_label']==-1].sort_values(by='subject_id'))
print('Training subjects')
print(subj_db[subj_db['gold_label']!=-1].sort_values(by='subject_id'))
'''
print('Classified Subjects:')
classified_subj_indx = np.where(subj_db['seen']>0)[0]
for indx in classified_subj_indx:
    print(subj_db.loc[indx]['subject_id'],subj_db.loc[indx]['history'][1:])

print('THRESHOLDS')
print(pd.DataFrame.from_dict(read_sqlite(file_path)['thresholds']))
print('CONFIG')
print(pd.DataFrame.from_dict(read_sqlite(file_path)['config']))
print('Key error summary:')'''
key_error_summary_dict = {'0':'Error in caesar_receive',\
                          '1':'Error in caesar_receive',\
                          '2':"Error when looking for 'already seen' flag in Caesar Extractor",\
                          '3':'Error when looking for user-id in Caesar Extractor',
                          '4':'Error in sending subjects to panoptes for retirement',\
                          '5':'Error when trying to delete messages from SQS queue in Caesar Extractor',\
                          '6':'No error set for this index',\
                          '7':'Catch-all exception - could be anywhere!',\
                          '8':'User sees the same test subject again'}
for k_i in key_error_summary_dict.keys():
    print(k_i,key_error_summary_dict[k_i])
for line in open(Config().keyerror_list_path,'r'):
    print(line)
print(f'Total number of test-subject classifications: {np.sum(subj_db[subj_db["gold_label"]==-1]["seen"])}')
print(f'Total number of unique test-subjects classified: {len(set((subj_db[subj_db["gold_label"]==-1]["subject_id"])))}')
print(f'Total number of training-subject classifications: {np.sum(subj_db[subj_db["gold_label"]!=-1]["seen"])}')
print(f'Max N-seen for a given subject: {np.max(subj_db["seen"])}')
print(f'Number of subjects seen >=4 times: {np.sum(subj_db["seen"]>=4)}')
print('Subjects seen >= 4 times')
print(subj_db[subj_db["seen"]>=4])
print(f'Number retired: {np.sum(subj_db["retired"])}')
known_subj_id = [91831901, 91831911, 91831916, 91831928, 91831943, 91831945, 91831956, 91831965, 91831970, 91831982, 91831993, 91832039, 91832118, 91832123, 91832163, 91832189, 91832209, 91832222, 91832261, 91832278, 91832315, 91832316, 91832331, 91832337, 91832344, 91832354, 91832394]
print(subj_db[(subj_db['subject_id'].isin(known_subj_id))&(subj_db['score']>0.9)])
'''
nohup python3 run.py > run_swap_log.log 2>&1 &
echo $! > process_id_number.txt
top -pid `cat process_id_number.txt`
###kill -9 `cat process_id_number.txt`
'''

#Finding known lenses from subject export:
'''
db = pd.read_csv('/Users/hollowayp/Downloads/space-warps-des-vision-transformer-subjects (5).csv')
db=db[db['workflow_id']==25011].reset_index()
known_len_subj_id = []
for subj_i in range(len(db)):
  meta_i = eval(db['metadata'][subj_i])
  if meta_i['#Type']=='SUB' and meta_i["#KNOWN_LENS_REF"]!="":
    print(meta_i["#KNOWN_LENS_REF"])
    known_len_subj_id.append(db['subject_id'][subj_i])
'''