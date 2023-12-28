"""
A basic script to create a golds csv from a set of classifications
"""
import argparse
import pandas as pd
import numpy as np
from collections import Counter
from config import Config
subjects_path = '/mnt/zfsusers/hollowayp/kSWAP/examples/data/space-warps-des-vision-transformer-subjects_Oct_2023.csv'
out_path = '/mnt/zfsusers/hollowayp/kSWAP/examples/data/DES_VT_Oct_2023.csv'
df = pd.read_csv(subjects_path)
print('ONLY PARSING GOLDS WHICH BELONG TO THE WORKFLOW: ',Config().workflow)
subject_sets = [116378,116379,116447,116380]
print(f'ONLY PARSING GOLDS WHICH BELONG TO THE SUBJECT SETS: {subject_sets}')
df = df[df['workflow_id']==Config().workflow]
# extract subject_id and gold status. Purge if not clear
subjects = []
skip_subjects = []
golds = []
unique_ids = set(df['subject_id'])
counter_dict = Counter(df['subject_id'])
duplication_list = [subj_repeat for subj_repeat in counter_dict if counter_dict[subj_repeat]>1]
if len(duplication_list)>0:
    print("WARNING: Duplicated subject id's are present in the subject database. Fix the duplication before continuing:", duplication_list)
types = ['SUB', 'DUD', 'LENS']
for row in df.iterrows():
    subject = row[1]['subject_id']
#NOTE IN SOME CLASSIFICATION FILES (NOT THIS ONE), WHETHER THE IMAGE IS A LENS/NOT_LENS ETC IS NOT IN THE METADATA COLUMN BUT ELSEWHERE.
    data = eval(row[1]['metadata'])
    if row[1]['subject_set_id'] not in subject_sets: continue
    if subject in subjects:
        continue
    elif subject in skip_subjects:
        continue
    else:
        try: obj_type = data['#feedback_1_type']
        except Exception as ex:
            skip_subjects.append(subject)
            continue
        if 'SUB' in obj_type:
            skip_subjects.append(subject)
        elif 'DUD' in obj_type:
            subjects.append(subject)
            golds.append(0)
        elif 'LENS' in obj_type:
            subjects.append(subject)
            golds.append(1)

golds=np.array(golds)
print(golds,np.sum(golds==0))
print(f'In total, have {np.sum(golds)} sims and {np.sum(golds==0)} duds, along with {len(skip_subjects)} test subjects')
for gold_subj in subjects:
    assert gold_subj not in skip_subjects
for test_subj in skip_subjects:
    assert test_subj not in subjects
golds = pd.DataFrame({'subject_id': subjects, 'gold': golds})
golds.to_csv(out_path, index=False)
