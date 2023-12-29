from pandas import read_sql_query, read_sql_table
import matplotlib.pyplot as pl
from tqdm import tqdm
import pandas as pd
import numpy as np
import statistics
import datetime
import sqlite3
import sys
sys.path.append('../kswap/')
from config import Config

def set_max_pd(rows=None,columns=None):
    pd.set_option('display.max_columns', columns)
    pd.set_option('display.max_rows', rows)

set_max_pd(rows=150,columns=None)

file_path = Config().examples_path+'/data/'+Config().db_name
print('filepath',file_path)

def thresholds_setting():
    sys.path.insert(0, '../kswap')
    p_real = Config().thresholds[1]
    p_bogus = Config().thresholds[0]
    return [p_real,p_bogus]

def prior_setting():
    return Config().p0

def read_sqlite(dbfile):
    with sqlite3.connect(dbfile) as dbcon:
      tables = list(read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
      out = {tbl : read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}
    return out

print('SUBJECTS')
subj_db = pd.DataFrame.from_dict(read_sqlite(file_path)['subjects'])
subj_db['score']= subj_db['score'].astype('float64')
print(f'Have {len(subj_db[subj_db["gold_label"]==1])} sims and {len(subj_db[subj_db["gold_label"]==0])} in the training set')
print(f"{len(set(subj_db[subj_db['gold_label']==-1]['subject_id']))} test subjects have been classified so far, of which"+\
      f"{len(subj_db[(subj_db['retired']==0)&(subj_db['gold_label']==-1)])} have not yet been retired")

print('Training subjects')
print(subj_db[subj_db['gold_label']!=-1].sort_values(by='subject_id'))

print('Test subjects')
Test_subj = subj_db[subj_db['gold_label']==-1].copy().reset_index(drop=True) 
print(Test_subj)

print('Retired Subjects')
print(subj_db[subj_db['retired']==1].sort_values(by='subject_id'))

print('USERS')
user_db =pd.DataFrame.from_dict(read_sqlite(file_path)['users']) 
N_subj_seen_by_user = [len(eval(user_db['user_subject_history'][i])) for i in range(len(user_db))]
user_db['N_subj_seen'] = N_subj_seen_by_user
print('User db')
print(user_db[['user_id', 'user_score', 'confusion_matrix','N_subj_seen']].sort_values(by='N_subj_seen'))
print("User_id's with >1000 classifications:",list(user_db[user_db['N_subj_seen']>=1000]['user_id']))

#Histogram of N_seen
hist_dict_nseen = {'fill':False,'density':False,'bins':np.arange(-0.5,np.max(subj_db['seen'])+1.5,1)}
fig,ax = pl.subplots(1,3,figsize=(15,5))
ax[0].hist(subj_db[subj_db['gold_label']==-1]['seen'],**hist_dict_nseen,edgecolor='k')
ax[0].hist(subj_db[subj_db['gold_label']==0]['seen'],**hist_dict_nseen,edgecolor='red')
ax[0].hist(subj_db[subj_db['gold_label']==1]['seen'],**hist_dict_nseen,edgecolor='blue')
ax[0].set_xlabel('N. Classifications')
ax[0].set_ylabel('N. Subjects')
ax[0].legend(['Test','Training (Non-lens)','Training (Lens)'])

#Histogram of N_seen by user
ax[1].hist(user_db['N_subj_seen'],bins=np.arange(-0.5,np.max(user_db['N_subj_seen'])+1.5,1),edgecolor='k',fill=False)
ax[1].set_xlabel('N. Classifications')
ax[1].set_ylabel('N. Users')

print('Key error summary:')
key_error_summary_dict = {'0':'Error in caesar_receive',\
                          '1':'Error in caesar_receive',\
                          '2':"Error when looking for 'already seen' flag in Caesar Extractor",\
                          '3':'Error when looking for user-id in Caesar Extractor',
                          '4':'Error in sending subjects to panoptes for retirement',\
                          '5':'Error when trying to delete messages from SQS queue in Caesar Extractor',\
                          '6':'Error when saving retired list - retired_list path name may not be set?',\
                          '7':'Catch-all exception - could be anywhere!',\
                          '8':'User sees the same test subject again',\
                          '9':'Error when trying to remove duplicate classifications in caesar_extractor'}
for k_i in key_error_summary_dict.keys():
    print(k_i,key_error_summary_dict[k_i])
for line in open(Config().keyerror_list_path,'r'):
    key_error_list = eval(line)
    print('Error Occurrences: ',{elem:key_error_list[elem] for elem in range(len(key_error_list))})

print(f'Total number of test-subject classifications: {np.sum(subj_db[subj_db["gold_label"]==-1]["seen"])}')
print(f'Total number of unique test-subjects classified: {len(set((subj_db[subj_db["gold_label"]==-1]["subject_id"])))}')
print(f'Total number of training-subject classifications: {np.sum(subj_db[subj_db["gold_label"]!=-1]["seen"])}')
print(f'Max N. classifications for a given subject: {np.max(subj_db["seen"])}')
print(f'Number of subjects seen >=5 times: {np.sum(subj_db["seen"]>=5)}')
print(f'Number of subjects retired: {np.sum(subj_db["retired"])}')
print(f'Mean number of classifications per test subject: {np.round(np.mean(subj_db[subj_db["gold_label"]==-1]["seen"]),2)}')
print(f'Median number of classifications per test subject: {np.round(np.median(subj_db[subj_db["gold_label"]==-1]["seen"]),2)}')
print(f'Modal number of classifications per test subject: {np.round(statistics.mode(subj_db[subj_db["gold_label"]==-1]["seen"]),2)}')

#Histogram of subject scores:
dS = 0.5
hist_dict = {'bins':np.arange(-10,dS,dS),'density':False,'fill':False}
ax[2].hist(np.log10(subj_db[subj_db['gold_label']==-1]['score']),**hist_dict,edgecolor='k')
ax[2].hist(np.log10(subj_db[subj_db['gold_label']==0]['score']),**hist_dict,edgecolor='red')
ax[2].hist(np.log10(subj_db[subj_db['gold_label']==1]['score']),**hist_dict,edgecolor='blue')
ylim_hist = pl.ylim()
ax[2].plot(np.log10(np.array([5e-4,5e-4])),[ylim_hist[0],ylim_hist[1]],'--',c='grey')
ax[2].plot([-5,-5],[ylim_hist[0],ylim_hist[1]],'--',c='red')
ax[2].plot([0,0],[ylim_hist[0],ylim_hist[1]],'--',c='blue')
ax[2].set_ylim(ylim_hist)
ax[2].legend(['Test','Training (Non-lens)','Training (Lens)'])
ax[2].set_xlabel('Log(Score)')
ax[2].set_ylabel('Counts')
pl.tight_layout()
pl.show()