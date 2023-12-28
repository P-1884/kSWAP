from panoptes_client.panoptes import PanoptesAPIException
from panoptes_client import Subject as PanoptesSubject
from panoptes_client import Panoptes, Workflow
from panoptes_client import subject_set
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import sqlite3
import time
import json
import csv
import sys
import os
sys.path.append('/mnt/zfsusers/hollowayp/')
sys.path.append('/mnt/zfsusers/hollowayp/kSWAP/kswap')

from config import Config

workflow = Workflow.find(Config().workflow)

subject_export_path = '/mnt/zfsusers/hollowayp/kSWAP/examples/data/space-warps-des-vision-transformer-subjects.csv'
print(f'Using data export: {subject_export_path}')
df = pd.read_csv(subject_export_path)


#Some of these subjects may be in a different workflow, hence cannot be unretired:
for subj_i in tqdm(df['subject_id']):
    try: workflow.unretire_subjects([subj_i])
    except StopIteration: print(f'Cannot unretire subject: {subj_i}')
    try: assert workflow.subject_workflow_status(subj_i).retirement_reason is None
    except StopIteration: print(f'Cannot find subject status of: {subj_i}')
    except AssertionError: print(f'ASSERTION ERROR {subj_i}')

retired_list = [92365379,92365224,92365707,92365813,92365051,92364930,92365382,92365545,92365291,92365391,92364951,92365432,92364856,92365672,92365195]
for subj_i in tqdm(retired_list):
    try: workflow.unretire_subjects([subj_i])
    except StopIteration: print(f'Cannot unretire subject: {subj_i}')
    try: assert workflow.subject_workflow_status(subj_i).retirement_reason is None
    except StopIteration: print(f'Cannot find subject status of: {subj_i}')
    except AssertionError as ex: 
        print(ex,workflow.subject_workflow_status(subj_i).retirement_reason)

from panoptes_client import Panoptes, Project, SubjectSet, Subject
