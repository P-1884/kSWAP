from collections import Counter
from datetime import datetime
import caesar_external as ce
from config import Config
from tqdm import tqdm
import pandas as pd
import numpy as np
import sqlite3
import logging
import random
import time
import json
import csv
import sys
import os
from panoptes_client import Panoptes, Workflow, subject_set
from panoptes_client import Subject as PanoptesSubject
from panoptes_client.panoptes import PanoptesAPIException

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

total_ignored = 0
total_not_ignored = 0

#Classification Class:
#Arguments: Classification_id, User_id, Subject_id, Annotation, Label_map
class Classification(object):
  def __init__(self,
               id,
               user_id,
               subject_id,
               annotation,
               label_map,
               classification_time):
      
    self.id = int(id)
    try:
      self.user_id = int(user_id)
    except ValueError:
      self.user_id = user_id
    self.subject_id = int(subject_id)
    self.label_map  = label_map
    self.label = self.parse(annotation)
    self.classification_time=classification_time

  def parse(self, annotation):
    try:
        value = annotation['T1'][0]['value']
    except:
        value = annotation[0]['value']
    if value==[]:
      return '0'
    else:
      return '1'

class User(object):
  def __init__(self,
               user_id,
               classes,
               gamma,
               user_default = None #i.e. initial user skill
               ):

    self.user_id = user_id
    self.classes = classes
    self.k = len(classes)
    self.gamma = gamma
    if user_default:
      self.user_score = user_default
    else:
      self.initialise_user_score()
    self.initialise_confusion_matrix()
    self.history = [('_', self.user_score,'_','_')]
    self.user_subject_history = []
  
  def initialise_confusion_matrix(self):
      '''
      Initialised to {N_seen: [0,0], N_gold: [0,0]}
      '''
      self.confusion_matrix = {'n_seen': [0]*self.k, 'n_gold': [0]*self.k}
  
  def initialise_user_score(self):
    '''
    Initialsed to [0.5,0.5] for binary classification.
    '''
    self.user_score = {}
    for i in range(self.k):
      self.user_score[self.classes[i]] = 1 / float(self.k)

  def update_confusion_matrix(self, gold_label, label):
    '''
    Adds 1 to 'n_gold' category for each subject of a particular type (lens or not-lens) seen.
    If correctly identified, adds 1 to 'n_seen' category for the subject.
    Note, names are a misnomer:
    - N_gold really means 'Number of each category seen'
    - N_seen really means 'Number of each category correctly identified'. 
    The user skill is not updated for non-gold (i.e. test) subjects.
    '''
    label=int(label)
    confusion_matrix = self.confusion_matrix
    confusion_matrix['n_gold'][gold_label] += 1
    if str(gold_label) == str(label):
      confusion_matrix['n_seen'][label] += 1
    self.confusion_matrix = confusion_matrix

  def update_user_score(self, gold_label, label):
    '''
    First updates the confusion matrix (above). Then updates the user skill via:
      (1+N_in_category_correctly_identified)/(2+N_in_category_seen) for gamma=1.
    The user skill is not updated for non-gold subjects.'''

    self.update_confusion_matrix(gold_label, label)
    try:
      score0 = (self.confusion_matrix['n_seen'][0] + self.gamma) \
             / (self.confusion_matrix['n_gold'][0] + 2.0*self.gamma)
    except ZeroDivisionError:
      score0 = self.user_default[self.classes[0]]
    try:
      score1 = (self.confusion_matrix['n_seen'][1] + self.gamma) \
             / (self.confusion_matrix['n_gold'][1] + 2.0*self.gamma)
    except ZeroDivisionError:
      score1 = self.user_default[self.classes[1]]
    self.user_score = {list(self.classes)[0]: score0, list(self.classes)[1]: score1}

  def dump(self):
    '''
    Returns user_id and then jsons of user score, confusion matrix and history for that user.
    '''
    return (self.user_id,
            json.dumps(self.user_score),
            json.dumps(self.confusion_matrix),
            json.dumps(self.history),
            json.dumps(self.user_subject_history))

class Subject(object):
  def __init__(self,
               subject_id,
               p0,
               classes,
               gold_label=-1,
               epsilon=1e-9,
               hard_sim_label = 0):
      
    self.subject_id = subject_id
    self.score = float(p0)
    self.classes = classes
    self.k = len(self.classes)
    self.gold_label = gold_label #Training dud = 0, Training sim = 1, Test subject = -1.
    self.epsilon = epsilon
    self.retired = False
    self.retired_as = None
    self.seen = 0 #Number of times the subject has been classified.
    '''
    The history attribute stores the following:
    [user_id, user_score, user classification, subject score, time_stamp, classification_time]
    where:
    'time_stamp' is the time at which the classification was processed by swap. 
    'classification_time' is the time recorded by Zooniverse as the time the classification was made.'''
    self.history = [('_', '_', '_', self.score,'_','_')]
    self.hard_sim_label = hard_sim_label #= 1 if subject is a hard sim, otherwise =0.

  def update_score(self, label, user,time_stamp,classification_time):
    '''
    This function:
    - Updates the subject score according to the user's classification and skill. See Marshall et al. 2015 
    for details (https://arxiv.org/pdf/1504.06148.pdf).
    - Increases seen by 1
    - Updates the subject history.
    Defining u_L and u_NL as user scores for Lens/Non-lens classification):
    If labelled NL:
    New score = score*(1-u_L)/[score*(1-u_L)+u_NL*(1-score)]
    If labelled L:
    New score = score*u_L/[score*u_L+(1-u_NL)*(1-score)]
    '''
    if label == '0':
      self.score=float(self.score)
      self.score = ((self.score) \
                 * (1 - user.user_score[list(self.classes)[1]])) \
                 / (self.score \
                 * (1 - user.user_score[list(self.classes)[1]]) \
                 + user.user_score[list(self.classes)[0]] \
                 * (1-self.score) \
                 + self.epsilon)
    elif label == '1':
      self.score=float(self.score)
      self.score = ((self.score) \
                 * user.user_score[list(self.classes)[1]]) \
                 / (self.score \
                 * user.user_score[list(self.classes)[1]] \
                 + (1-user.user_score[list(self.classes)[0]]) \
                 * (1-self.score)
                 + self.epsilon)
    #Update the subject history:
    self.history.append((user.user_id, user.user_score, label, self.score, time_stamp, classification_time))
    self.seen += 1

  def dump(self):
    #Returns subject_id, gold_label, retired status/classification, seen, then jsons of history and score.
    return (self.subject_id, \
            self.gold_label, \
            json.dumps(self.score), \
            self.retired, \
            self.retired_as, \
            self.seen, \
            json.dumps(self.history),\
            self.hard_sim_label)

def find_subjects(subject_id):
    return PanoptesSubject().find(subject_id)

class SWAP(object):
  def __init__(self,
               config=None,
               timeout=10):
      
    self.users = {}
    self.subjects = {}
    self.objects = {}
    self.config=config
    self.last_id = 0
    self.seen = set([])
    self.db_exists = False
    self.instance_counts = [0,0,0,0,0,0]
    self.timeout = timeout # wait x seconds to acquire db connection
    try:
      self.create_db()
      self.save()
    except sqlite3.OperationalError:
      self.db_exists = True

    Panoptes.connect(username='USERNAME', \
                     password='PASSWORD')
                     
    self.workflow = Workflow.find(config.workflow)
    
  def connect_db(self):
    return sqlite3.connect(self.config.db_path+self.config.db_name, timeout=self.timeout)

  def create_db(self):
    '''
    Creates a database in which the subject, user and configuration data is stored.
    '''
    conn = self.connect_db()
    conn.execute('CREATE TABLE users (user_id PRIMARY KEY, user_score, ' +\
                 'confusion_matrix, history, user_subject_history)')

    conn.execute('CREATE TABLE subjects (subject_id PRIMARY KEY, ' +\
                 'gold_label, score, retired, retired_as, seen ,history, hard_sim_label)')
                 
    conn.execute('CREATE TABLE thresholds (thresholds)')

    conn.execute('CREATE TABLE config (id PRIMARY KEY, user_default, ' +\
                 'workflow, p0, gamma, retirement_limit, db_path, ' +\
                 'db_name, timeout, last_id, seen)')

    conn.close()

  def load_users(self, users):
    '''
    For all the users in the user dictionary:
    1) Loads their scores,
    2) Sets all of them to be members of the User class, with their respective values for history etc.'''
    for user in users:
      user_score = json.loads(user['user_score'])
      self.users[user['user_id']] = User(user_id=user['user_id'],
                                         classes=self.config.label_map.keys(),
                                         gamma=self.config.gamma,
                                         user_default=user_score)
      self.users[user['user_id']].confusion_matrix = json.loads(user['confusion_matrix'])
      self.users[user['user_id']].history = json.loads(user['history'])
      self.users[user['user_id']].user_subject_history = json.loads(user['user_subject_history'])

  def load_subjects(self, subjects):
    '''As above, but for all subjects in the subject dictionary.'''
    for subject in subjects:
      self.subjects[subject['subject_id']] = Subject(subject_id=subject['subject_id'],
                                                     classes=self.config.label_map.keys(),
                                                     p0=self.config.p0)
      self.subjects[subject['subject_id']].score = float(subject['score'])
      self.subjects[subject['subject_id']].gold_label = subject['gold_label']
      self.subjects[subject['subject_id']].retired = subject['retired']
      self.subjects[subject['subject_id']].retired_as = subject['retired_as']
      self.subjects[subject['subject_id']].seen = subject['seen']
      self.subjects[subject['subject_id']].history = json.loads(subject['history'])
      self.subjects[subject['subject_id']].hard_sim_label = subject['hard_sim_label']

  def load(self):
    '''Fetches all the data from the databases made above, then makes them members of the users/subjects classes.'''
    def it(rows):
      for item in rows:
        yield dict(item)
    
    conn = self.connect_db()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute('SELECT * FROM config')
    config = dict(c.fetchone())
                 
    swap = SWAP(config=self.config,
                timeout=config['timeout'])
    swap.last_id=self.last_id
    swap.seen = set(json.loads(config['seen']))
  
    c.execute('SELECT * FROM users')
    swap.load_users(it(c.fetchall()))
    c.execute('SELECT * FROM subjects')
    swap.load_subjects(it(c.fetchall()))

    conn.close()

    return swap

  def dump_users(self):
    '''Appends all the users to a list then returns'''
    users = []
    for u in self.users.keys():
      users.append(self.users[u].dump())
    return users

  def dump_subjects(self):
    '''As above for subjects'''
    subjects = []
    for s in self.subjects.keys():
      subjects.append(self.subjects[s].dump())
    return subjects

  def dump_objects(self):
    '''As above for objects'''
    objects = []
    for o in self.objects.keys():
      objects.append(self.objects[o].dump())
    return objects

  def dump_config(self):
    '''#As above for the config data.'''
    return (0, json.dumps(self.config.user_default), self.config.workflow,
            self.config.p0, self.config.gamma, self.config.retirement_limit,
            self.config.db_path, self.config.db_name, self.timeout,
            self.last_id, json.dumps(list(self.seen)))

  def save(self):
    '''Adds/saves data into databases.'''
    conn = self.connect_db()
    c = conn.cursor()
    def zip_name(data):
      return [d.values() for d in data]
    c.executemany('INSERT OR REPLACE INTO users VALUES (?,?,?,?,?)',
                   self.dump_users())
    c.executemany('INSERT OR REPLACE INTO subjects VALUES (?,?,?,?,?,?,?,?)',
                   self.dump_subjects())                
    c.execute('INSERT OR REPLACE INTO config VALUES (?,?,?,?,?,?,?,?,?,?,?)',
               self.dump_config())
    conn.commit()
    conn.close()

  def process_classification(self, cl, online=False, from_csv = False):
    '''
    This function does the following:
    1) Checks if classification is new (if not, function does nothing)
    2) Checks if the user is new - if so, makes them a member of User class
    3) Checks if the subject is new - if so, makes them a member of Subject class.
    Then:
    4) Updates subject score due to this classification. 
    5) If subject is a gold and swap being run in online mode, it also updates user score and adds it to user history.'''
    # check if user is known
    time_stamp = time.time()
    classification_time = cl.classification_time
    try:
      self.users[cl.user_id]
    except KeyError:
      self.users[cl.user_id] = User(user_id = cl.user_id,
                                    classes = self.config.label_map.keys(),
                                    gamma   = self.config.gamma,
                                    user_default = self.config.user_default)
    # check if subject is known
    try:
      self.subjects[cl.subject_id]
    except KeyError:
      self.subjects[cl.subject_id] = Subject(subject_id = cl.subject_id,
                                             p0 = self.config.p0,
                                             classes = self.config.label_map.keys())
    '''
    If the subject has not previously been seen by the user before, the user skill/subject scores are updated.
    Otherwise, the classification is ignored.
    '''
    if cl.subject_id not in self.users[cl.user_id].user_subject_history:
        self.subjects[cl.subject_id].update_score(cl.label, self.users[cl.user_id],time_stamp,classification_time)
        if online:
          if self.subjects[cl.subject_id].gold_label==0:
            self.users[cl.user_id].update_user_score(0, cl.label)
          if self.subjects[cl.subject_id].gold_label==1:
            '''If the subject is:
            2) An easy sim or
            1) A hard sim and the user correctly classified it
            The user skill is updated.'''
            if self.subjects[cl.subject_id].hard_sim_label == 0 or (self.subjects[cl.subject_id].hard_sim_label == 1 and int(cl.label)==1):
              self.users[cl.user_id].update_user_score(1, cl.label)
            else:
              '''
              If the subject is a hard sim and the user mis-classifies it, the user skill is not updated.'''
              print('Hard sim: not updating user score')
        self.users[cl.user_id].history.append((cl.subject_id, \
                                               self.users[cl.user_id].user_score,\
                                               time_stamp,classification_time))
        self.users[cl.user_id].user_subject_history.append(cl.subject_id)
        self.last_id = cl.id
        self.seen.add(cl.id)
        global total_not_ignored
        total_not_ignored+=1
    else:
        print('User has seen subject before: Ignoring')
        global total_ignored
        total_ignored+=1
        print('Total ignored so far: ',total_ignored)
        if self.subjects[cl.subject_id].gold_label==-1:
            #Users should not be shown test subjects again - Key Error 8 indicates if they have been.
            print("KEY_ERROR_8 HERE")
            with open(self.config.keyerror_list_path,'r') as g:
                keyerror_list=eval(g.read())
                keyerror_list[8]+=1
            with open(self.config.keyerror_list_path,'w') as q:
                q.write(str(keyerror_list))

  def retire(self, subject_batch):
    '''
    Function to check if a subject needs retiring due to score thresholds.
    - Gold (i.e. training) subjects are not retired.
    - Test subjects are retired if their score is beyond either threshold and they have received >= the
    classification lower limit. If so, they are added to the
    'to_retire' list and marked as '0' or '1' depending on which threshold (lower or upper) they passed.
    '''
    to_retire = []
    for subject_id in subject_batch:
      subject = self.subjects[subject_id] #Have removed a 'try' 'except' clause from here since the DES-VT run.
      if subject.gold_label in (0,1):
        print('Not retiring as subject is gold')
        continue
      subject.score=float(subject.score)
      if (subject.score < self.config.thresholds[0]) and subject.seen>=self.config.lower_retirement_limit:
        assert subject.gold_label !=-1
        subject.retired = True
        self.subjects[subject_id].retired=True
        subject.retired_as = 0
        to_retire.append(subject_id)
      elif subject.score > self.config.thresholds[1] and subject.seen>=self.config.lower_retirement_limit:
        subject.retired = True
        self.subjects[subject_id].retired=True
        subject.retired_as = 1
        to_retire.append(subject_id)
    return to_retire #Needs to return a list.

  def retire_classification_count(self, subject_batch):
    '''
    Function to check if a subject needs retiring due to reaching the upper classification limit.
    - Gold (i.e. training) subjects are not retired.
    - Test subjects are retired with their final class based on the majority vote of its previous 
    classifications, **not** a rounded version of its current score.
    '''
    def majority_vote(sequence):
      occurence_count = Counter(sequence)
      return occurence_count.most_common(1)[0][0]
    to_retire = []
    for subject_id in subject_batch:
      subject = self.subjects[subject_id] #Have removed a 'try' 'except' clause from here since the DES-VT run.
      if subject.seen >= self.config.retirement_limit and subject.gold_label==-1:
        subject.retired = True
        self.subjects[subject_id].retired=True
        subject.retired_as = majority_vote([h[2] for h in subject.history[1:len(subject.history)]])
        to_retire.append(subject_id)
    return to_retire #Needs to return a list.

  def send_panoptes(self, subject_batch,reason):
    '''
    Sends the subjects to be retired to panoptes.'''
    subjects = []
    try:
        print('Retiring the following subjects via panoptes',subject_batch)
        self.workflow.retire_subjects([str(s_i) for s_i in subject_batch], reason = reason)
        for subject_i in subject_batch:
            print(subject_i,self.workflow.subject_workflow_status([subject_i]).retirement_reason)
    except Exception as excep:
        '''Key Error 4 indicates if there is a problem with panoptes retirement'''
        print(excep)
        print('KEY_ERROR_4 HERE')
        with open(self.config.keyerror_list_path,'r') as g:
            keyerror_list=eval(g.read())
            keyerror_list[4]+=1
        with open(self.config.keyerror_list_path,'w') as q:
            q.write(str(keyerror_list))

  def retrieve_list(self,list_path):
    '''
    Function to retrieve various lists which are generated/updated while SWAP runs (e.g. key_error_list)
    '''
    try:
        df = pd.read_csv(list_path)
        list_total = []
        for row in df.iterrows():
            try:
              list_i = eval(row[1]['Column 1'])
            except:
              list_i=row[1]['Column 1']
            list_total.append(list_i)
        return list_total
    except FileNotFoundError:
        print('Error with file retrieval')
        return []

  def save_list(self,list_total,list_path):
    list_db = pd.DataFrame({'Column 1': list_total}) #Dont change this column title unless change those in def retrieve_list() too.
    list_db.to_csv(list_path, index=False)

  def process_classifications_from_csv_dump(self, path, online=False):
    '''
    This function goes through each row in the csv, defining classification events, updating scores 
    (as per process_classification) and labelling subjects to be retired if required.'''
    with open(path, 'r',encoding='utf-8-sig') as csvdump:
      reader = csv.DictReader(csvdump)
      for row in reader:
        id = int(row['classification_id'])
        classification_time=row['created_at']
        try:
          assert int(row['workflow_id']) == self.config.workflow
        except AssertionError as e:
           print(int(row['workflow_id']),'Classification not from the specified workflow: Ignoring this classification')
           continue
        try:
          user_id = int(row['user_id'])
        #To ignore not-logged-on users, uncomment the two lines below:
        #if str(row['user_id'][0:13])=='not-logged-in':
        #  continue
        except ValueError:
          user_id = row['user_name']
        #To ignore not-logged-on users, uncomment the two lines below:
        #if str(row['user_name'][0:13])=='not-logged-in':
        #  continue
        dt_row = row['created_at']
        subject_id = int(row['subject_ids'])
        annotation = json.loads(row['annotations'])
        try:
          cl = Classification(id,
                              user_id,
                              subject_id,
                              annotation,
                              label_map=self.config.label_map,
                              classification_time=classification_time)
        except ValueError:
            continue
        self.process_classification(cl, online,from_csv=True)
    '''
    #If you want subjects to be sent for retirement after processing the csv file, uncomment these lines:
    retired_list_threshold_updates = self.retire(self.subjects.keys())
    retired_list_Nclass_updates = self.retire_classification_count(self.subjects.keys())
    self.send_panoptes(retired_list_threshold_updates,'consensus')
    self.send_panoptes(retired_list_Nclass_updates,'classification count')
    #
    retired_list = self.retrieve_list(self.config.retired_items_path)
    retired_list.extend(list(set(np.array(retired_list_threshold_updates+retired_list_Nclass_updates))))
    self.save_list(retired_list,self.config.retired_items_path)
    #
    retired_list_Nclass = self.retrieve_list(self.config.Nclass_retirement_path)
    retired_list_Nclass.append([time.time(),retired_list_Nclass_updates])
    self.save_list(retired_list_Nclass,self.config.Nclass_retirement_path)
    #
    retired_list_Nthres = self.retrieve_list(self.config.Nthres_retirement_path)
    retired_list_Nthres.append([time.time(),retired_list_threshold_updates])
    self.save_list(retired_list_Nthres,self.config.Nthres_retirement_path)'''

  def caesar_recieve(self, ce):
    '''
    Recieves data from caesar and processes classifications (as per process_classification)'''
    keyerror_1_i=0;keyerror_2_i=0
    try:
        st = time.time()
        data = ce.SQSExtractor.sqs_retrieve_fast()
        haveItems = False
        subject_batch = []
        q_all=0
        print(f'Beginning processing of {len(data)} subjects')
        for i in range(len(data)):
          item = data[i]
          try:
              q_all+=1
              haveItems = True
              id = int(item['id'])
              try:
                user_id = int(item['user'])
              except ValueError:
                user_id = item['user']
              #Have removed a 'try/except' for TypeError's here.
              subject_id = int(item['subject'])
              classification_time=item['classification_time']
              annotation = item['annotations']
              cl = Classification(id, user_id, subject_id, annotation,label_map=self.config.label_map,classification_time=classification_time)
              self.process_classification(cl,online=True)
              self.last_id = id
              self.seen.add(id)
              subject_batch.append(subject_id)
          except KeyError as e:
            '''Key Error 0 indicates if there is a problem with retrieving/processing a particular subject via caesar'''
            print('KEY_ERROR_0 HERE',e)
            keyerror_1_i+=1
            continue
        et=time.time()
        if q_all!=0:
            logging.info('time taken = ' + str(et-st)+ ' for ' + str(q_all) + ' classifications, ' + str((et-st)/q_all) + 's/cl')
        return haveItems, subject_batch,keyerror_1_i,keyerror_2_i, q_all
    except Exception as ex:
        '''Key Error 1 indicates if there is a problem with retrieving/processing subjects via caesar'''
        print('KEY_ERROR_1 HERE',ex)
        keyerror_2_i+=1
        return False,[],0,0,keyerror_1_i,keyerror_2_i,0

  def update_retirement_lists(self,retire_batch,retire_list_Nclass,retire_list_thres):
    '''
    This saves the ID's of subjects which have been retired, along with the retirement time.
    '''
    print(f'Retired {retire_list_Nclass} due to classification limit and {retire_list_thres} due to score thresholds')
    if len(retire_batch)>0:
      #Have removed a try/except clause here:
      if len(retire_list_Nclass)>0:
        retired_list = self.retrieve_list(self.config.Nclass_retirement_path)
        retired_list.append([time.time(),retire_list_Nclass])
        self.save_list(retired_list,self.config.Nclass_retirement_path)
      if len(retire_list_thres)>0:
        retired_list = self.retrieve_list(self.config.Nthres_retirement_path)
        retired_list.append([time.time(),retire_list_thres])
        self.save_list(retired_list,self.config.Nthres_retirement_path)  

  def process_classifications_from_caesar(self, caesar_config_name):
    '''
    This is the main function for AWS swap. It runs caesar_receive (which retrieves classifications from the queue,
    and updates user skills/subject scores), then retires subjects as necessary, in real time. The database is saved
    every 30 minutes (this can be changed below).
    '''
    ce.Config.load(caesar_config_name)
    with open(self.config.aws_list_path,'r') as f:
        aws_list=eval(f.read())
    k=0
    save_timer_start = time.time()
    try:
      while True:
        ST_time = time.time()
        haveItems, subject_batch,keyerror_1_i,keyerror_2_i,N_proc = self.caesar_recieve(ce)
        try:
          aws_list[N_proc]+=1 #= Number of times a retrieval has been made containing N_proc messages.
        except:
          pass
        if haveItems:
          k=0
          print(aws_list)
          st_save=time.time()
          ce.Config.instance().save()
          # load the just saved ce config
          ce.Config.load(caesar_config_name)
          et_save=time.time()
          print('Save time: ' + str(et_save-st_save))
        else:
          k+=1
          print(aws_list)
        retire_list_thres=self.retire(subject_batch)
        retire_list_Nclass=self.retire_classification_count(subject_batch)
        retire_batch = list(set(np.array(retire_list_thres+retire_list_Nclass)))
        logging.info('Retiring ' + str(len(retire_batch))+' subjects: ' + str(retire_batch))
        try:
            retired_list = self.retrieve_list(self.config.retired_items_path)
            retired_list.extend(retire_batch)
            self.save_list(retired_list,self.config.retired_items_path)
        except Exception as ex:
            '''Key Error 6 indicates if there is a problem with saving the retirement list'''
            with open(self.config.keyerror_list_path, 'r') as g:
                keyerror_list=eval(g.read())
            print('KEY_ERROR_6 HERE',ex)
            keyerror_list[6]+=1
            with open(self.config.keyerror_list_path, 'w') as h:
                h.write(str(keyerror_list))
        ST_retirement=time.time()
        print('Sending to panoptes')
        self.send_panoptes(retire_batch,'consensus')
        print('Sent to panoptes')
        self.update_retirement_lists(retire_batch,retire_list_Nclass,retire_list_thres)
        ED_retirement=time.time()
        logging.info('Retirement time: ' + str((ED_retirement-ST_retirement)/np.max([1,len(retire_batch)])) + '/subj for ' +  str(len(retire_batch)) + ' retirements')
        #Saves database every 30 minutes:
        if time.time()-save_timer_start>1800:
          print('Starting saving')
          self.save()
          save_timer_start = time.time()
          print('Finished saving')
        if keyerror_1_i != 0 or keyerror_2_i !=0:
            with open(self.config.keyerror_list_path,'r') as g:
              keyerror_list=eval(g.read())
            keyerror_list[0]+=keyerror_1_i
            keyerror_list[1]+=keyerror_2_i
            with open(self.config.keyerror_list_path,'w') as q:
              q.write(str(keyerror_list))
        with open(self.config.aws_list_path,'w') as f:
          f.write(str(aws_list))
        print('')
        ED_time = time.time()
        print('Total loop time per subject:' + str((ED_time-ST_time)/np.max([N_proc,1])) +' for ' + str(N_proc) + ' classifications')
    except KeyboardInterrupt as e:
      print('Starting save')
      self.save()
      print('Finishing save')
      retire_list_thres = self.retire(subject_batch)
      retire_list_Nclass = self.retire_classification_count(subject_batch)
      retire_batch = list(set(np.array(retire_list_thres+retire_list_Nclass)))
      self.send_panoptes(retire_batch,'consensus')
      self.update_retirement_lists(retire_batch,retire_list_Nclass,retire_list_thres)
      with open(self.config.aws_list_path,'w') as f:
        f.write(str(aws_list))
      self.save()
      print('Received KeyBoardInterupt: Terminating SWAP instance.')
      exit()
    except Exception as exep1:
        '''
        Key Error 7 is a catch-all exception to save the database in case of major bugs.
        '''
        print('KEY_ERROR_7 HERE',exep1)
        with open(self.config.keyerror_list_path, 'r') as g:
            keyerror_list=eval(g.read())
        keyerror_list[7]+=1
        with open(self.config.keyerror_list_path, 'w') as h:
            h.write(str(keyerror_list)) 
        self.save()
    st=time.time()
    retire_batch = list(set(np.array(retire_list_thres+retire_list_Nclass)))
    self.send_panoptes(retire_batch,'consensus')
    self.update_retirement_lists(retire_batch,retire_list_Nclass,retire_list_thres)
    self.save()
    with open(self.config.aws_list_path,'w') as f:
      f.write(str(aws_list))
    et=time.time()
    logging.info('Retirement time: ' + str(et-st))

#Adds gold subjects from the csv to Subject class.
  def get_golds(self, path, hard_sims_path = None):
    '''
    This function adds gold (=training) subjects to the Subject class, and identifies hard-sims from the hard_sims csv 
    provided.
    '''
    N_g=0
    with open(path,'r') as csvdump:
      reader = csv.DictReader(csvdump)
      for row in reader:
        subject_id = int(row['subject_id'])
        gold_label = int(row['gold'])
        try:
          self.subjects[subject_id].gold_label=gold_label
        except KeyError:
          N_g+=1
          self.subjects[subject_id] = Subject(subject_id,
                                            classes = self.config.label_map.keys(),
                                            p0 = self.config.p0,
                                            gold_label = gold_label)
      print('Adding ' + str(N_g)+' new golds')
    HS=0
    if hard_sims_path!=None:
        with open(hard_sims_path,'r') as csvdump_hardsims:
          reader = csv.DictReader(csvdump_hardsims)
          for row in reader:
            subject_id = int(row['subject_id'])
            self.subjects[subject_id].hard_sim_label=1
            HS+=1
    print('Adding '+ str(HS) + ' hard sims')

  def apply_golds(self, path):
    '''
    This function goes through the rows of the csv file (provided by the 'path' argument): if the subject is a gold
    the user score is updated accordingly. This is used in "offline" swap mode.
    NOTE:  There is nothing yet preventing user scores being updated multiple times if a given user 
    saw the same gold subjects more than once in test_offline or test_online. This is resolved for AWS swap.
    '''
    with open(path, 'r') as csvdump:
      reader = csv.DictReader(csvdump)
      for row in reader:
        id = int(row['classification_id'])
        try:
          user_id = int(row['user_id'])
        except ValueError:
          user_id = row['user_name']
        subject_id = int(row['subject_ids'])
        annotation = json.loads(row['annotations'])
        classification_time=row['created_at']
        try:
          assert int(row['workflow_id']) == self.config.workflow
        except AssertionError as e:
          print('Assertion Error: Ignoring classifications from the wrong workflow')
          continue 
        cl = Classification(id,
                              user_id,
                              subject_id,
                              annotation,
                              label_map=self.config.label_map,
                              classification_time=classification_time)
        try:
          self.users[cl.user_id]
        except KeyError:
          self.users[cl.user_id] = User(user_id = cl.user_id,
                                        classes = self.config.label_map.keys(),
                                        gamma   = self.config.gamma,
                                        user_default = self.config.user_default)
        try:
          gold_label = self.subjects[cl.subject_id].gold_label
          '''
          User skill is updated if:
          1) The subject is an easy training subject or
          2) The subject is a hard-sim and the user correctly classifies it.
          '''
          if gold_label==0:
            self.users[cl.user_id].update_user_score(gold_label, cl.label)
          elif gold_label==1 and self.subjects[cl.subject_id].hard_sim_label==0:
            self.users[cl.user_id].update_user_score(gold_label, cl.label)
          elif gold_label==1 and self.subjects[cl.subject_id].hard_sim_label==1 and cl.label == gold_label:
            self.users[cl.user_id].update_user_score(gold_label, cl.label)
        except KeyError as e:
          print('KEY ERROR IN APPLY GOLDS.')
          continue

  def run_offline(self, gold_csv, classification_csv, hard_sims_csv=None):
    self.get_golds(gold_csv, hard_sims_csv)
    self.apply_golds(classification_csv)
    self.process_classifications_from_csv_dump(classification_csv)

  def run_online(self, gold_csv, classification_csv,hard_sims_csv=None):
    self.get_golds(gold_csv,hard_sims_csv)
    self.process_classifications_from_csv_dump(classification_csv, online=True)
    print('Total ignored:     ',total_ignored)
    print('Total not ignored: ',total_not_ignored)

  def run_caesar(self,gold_csv,hard_sims_csv=None):
    self.load()
    self.get_golds(gold_csv,hard_sims_csv)
    self.process_classifications_from_caesar('AWS_config')