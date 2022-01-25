"""
A basic script to create a golds csv from a set of classifications
"""
import argparse
import pandas as pd

subjects_path = '/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/beta_test_25jan_subjects.csv'
out_path = '/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/des_golds_beta_test_25jan.csv'

df = pd.read_csv(subjects_path)

# extract subject_id and gold status. Purge if not clear
subjects = []
skip_subjects = []
golds = []
unique_ids = set(df['subject_id'])

types = ['SUB', 'DUD', 'LENS']
for row in df.iterrows():
    subject = row[1]['subject_id']
#NOTE IN SOME CLASSIFICATION FILES (NOT THIS ONE), WHETHER THE IMAGE IS A LENS/NOT_LENS ETC IS NOT IN THE METADATA COLUMN BUT ELSEWHERE.
    data = row[1]['metadata']
    if subject in subjects:
        continue
    elif subject in skip_subjects:
        continue
    else:
        if 'SUB' in data:
            skip_subjects.append(subject)
        elif 'DUD' in data:
            subjects.append(subject)
            golds.append(0)
        elif 'LENS' in data:
            subjects.append(subject)
            golds.append(1)

golds = pd.DataFrame({'subject_id': subjects, 'gold': golds})
golds.to_csv(out_path, index=False)
