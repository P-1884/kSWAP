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

retired_list = []
for subj_i in tqdm(retired_list):
    try: workflow.unretire_subjects([subj_i])
    except StopIteration: print(f'Cannot unretire subject: {subj_i}')
    try: assert workflow.subject_workflow_status(subj_i).retirement_reason is None
    except StopIteration: print(f'Cannot find subject status of: {subj_i}')
    except AssertionError as ex: 
        print(ex,workflow.subject_workflow_status(subj_i).retirement_reason)

from panoptes_client import Panoptes, Project, SubjectSet, Subject
