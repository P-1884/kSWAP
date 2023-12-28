import os
import csv
import numpy as np
import json
import sqlite3
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
from config import Config
import time
from tqdm import tqdm
from collections import Counter
from datetime import datetime
import random
from panoptes_client import Panoptes, Workflow
from panoptes_client import Subject as PanoptesSubject
from panoptes_client.panoptes import PanoptesAPIException
from panoptes_client import subject_set
import sys
sys.path.append('/mnt/zfsusers/hollowayp/')
try:
    import caesar_external as ce
    print(print(ce.__file__))
except ModuleNotFoundError:
    pass
import time
import multiprocess as mp
from multiprocessing.dummy import Pool
import caesar_external as ce
total_ignored = 0
total_not_ignored = 0
#**See note line 459**
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

#User Class:
#Arguments; User_ID, Classes (0,1), Gamma (1), User_default (ie default score), k (ie N. classes).
#Confusion matrix: Initialised to {N_seen: [0,0], N_gold: [0,0]}
#User Score: Initialsed to [0.5,0.5]
#History: Initialised to [(_, [0.5,0.5])] ie [(subject_id, user_score)].

#update_confusion_matrix: Adds 1 to 'N_gold' category for the subject (ie NL, L). If correctly identified, adds 1 to 'N_seen' category for the subject.
#Note, names are a misnomer, N_gold really means 'Number of each category seen', and N_seen really means 'Number of each category correctly identified'. 
#Note: Cant update confusion matrix for non-gold subjects - golds are labelled 0 or 1 within csv. Not-golds aren't included.

#update_user_score: First updates the confusion matrix (above). Then updates scores for NL and L via: (1+N_category_correct_identified)/(2+N_category_seen).
#Note Cant update user score for non-gold subjects.

#dump: returns user_id, then jsons of user score, confusion matrix and history for that user.
class User(object):
  def __init__(self,
               user_id,
               classes,
               gamma,
               user_default=None):

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
      self.confusion_matrix = {'n_seen': [0]*self.k, 'n_gold': [0]*self.k}
  
  def initialise_user_score(self):
    self.user_score = {}
    for i in range(self.k):
      self.user_score[self.classes[i]] = 1 / float(self.k)

  def update_confusion_matrix(self, gold_label, label):
    label=int(label)
    confusion_matrix = self.confusion_matrix
    confusion_matrix['n_gold'][gold_label] += 1
    if str(gold_label) == str(label):
      confusion_matrix['n_seen'][label] += 1
    self.confusion_matrix = confusion_matrix

  def update_user_score(self, gold_label, label):
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
    return (self.user_id,
            json.dumps(self.user_score),
            json.dumps(self.confusion_matrix),
            json.dumps(self.history),
            json.dumps(self.user_subject_history))

#Subject Class:
#Arguments: Subject_ID, p0 (ie initial probability), classes (0,1) , gold-label (default= -1 ie not training), k (ie N. classes), retired (=False).
#Retired: Initialised to False
#History: Initialised as [_,_,_,p0] ie (user_id, user_score, label, subject_score)
#Seen: Initialised to 0

#update_score: Updates subject scores, increases seen by 1 and appends (user_id, user_score, label, subject_score) to history.
#Defining u_l and u_NL as user scores for L/NL classification):
#If labelled NL:
#Score =score*(1-u_L)/[score*(1-u_L)+u_NL*(1-score)]
#If labelled L:
#Score = score*u_L/[score*u_L+(1-u_NL)*(1-score)

#dump: Returns subject_id, gold_label, retired status/classification, seen, then jsons of history and score.
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
    self.gold_label = gold_label
    self.epsilon = epsilon
    self.retired = False
    self.retired_as = None
    self.seen = 0
    self.history = [('_', '_', '_', self.score,'_','_')]
    self.hard_sim_label = hard_sim_label

  def update_score(self, label, user,time_stamp,classification_time):
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
    self.history.append((user.user_id, user.user_score, label, self.score,time_stamp,classification_time))
    self.seen += 1

  def dump(self):
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

#Initialises dictionaries for users, subjects, objects (?), config etc. Creates database:
#users (feat: user_id, user_score, confusion_matrix and history),
#subjects(feat: subject_id, gold_label, score, retired status/classification, seen and history)
#thresholds (feat: thresholds)
#config (feat: id (?), user_default, workflow, p0, gamma, retirement limit, path to database, database name, timeout, last_id and seen).
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
    self.instance_counts=[0,0,0,0,0,0]
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
#For all the users in the user dictionary: 1) Loads their scores, 2) Sets all of them to be members of the User class, with their respective values for history etc.
  def load_users(self, users):
    for user in users:
      user_score = json.loads(user['user_score'])
      self.users[user['user_id']] = User(user_id=user['user_id'],
                                         classes=self.config.label_map.keys(),
                                         gamma=self.config.gamma,
                                         user_default=user_score)
      self.users[user['user_id']].confusion_matrix = json.loads(user['confusion_matrix'])
      self.users[user['user_id']].history = json.loads(user['history'])
      self.users[user['user_id']].user_subject_history = json.loads(user['user_subject_history'])

#As above, but for all subjects in the subject dictionary.
  def load_subjects(self, subjects):
    for subject in subjects:
      self.subjects[subject['subject_id']] = Subject(subject_id=subject['subject_id'],
                                                     classes=self.config.label_map.keys(),
                                                     p0=self.config.p0)
      self.subjects[subject['subject_id']].score = float(subject['score'])
      self.subjects[subject['subject_id']].gold_label = subject['gold_label']
      self.subjects[subject['subject_id']].retired = subject['retired']
      self.subjects[subject['subject_id']].retired_as = subject['retired_as']
#      print('loading subjects'+str(subject['retired_as']))
      self.subjects[subject['subject_id']].seen = subject['seen']
      self.subjects[subject['subject_id']].history = json.loads(subject['history'])
      self.subjects[subject['subject_id']].hard_sim_label = subject['hard_sim_label']

#Fetches all the data from the databases made above, then makes them members of the users/subjects classes.
  def load(self):
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
#Appends all the users to a list then returns
  def dump_users(self):
    users = []
    for u in self.users.keys():
      users.append(self.users[u].dump())
    return users
#As above for subjects
  def dump_subjects(self):
    subjects = []
    for s in self.subjects.keys():
      subjects.append(self.subjects[s].dump())
    return subjects
#As above for objects
  def dump_objects(self):
    objects = []
    for o in self.objects.keys():
      objects.append(self.objects[o].dump())
    return objects
#As above for the config data.
  def dump_config(self):
    return (0, json.dumps(self.config.user_default), self.config.workflow,
            self.config.p0, self.config.gamma, self.config.retirement_limit,
            self.config.db_path, self.config.db_name, self.timeout,
            self.last_id, json.dumps(list(self.seen)))

#Adds data into databases.
  def save(self):
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

#Checks if classification is new (if not, function does nothing), if the user is new (if so, makes them a member of User class), if subject is new (if so, makes them a member of Subject class).
#Then: Updates subject score due to this classification. If subject is a gold and swap being run in online mode, it also updates user score and adds it to user history.
  def process_classification(self, cl, online=False, from_csv = False):
    # check user is known
    time_stamp = time.time()
    classification_time=cl.classification_time
    try:
      self.users[cl.user_id]
    except KeyError:
      self.users[cl.user_id] = User(user_id = cl.user_id,
                                    classes = self.config.label_map.keys(),
                                    gamma   = self.config.gamma,
                                    user_default = self.config.user_default)
    # check subject is known
    try:
      self.subjects[cl.subject_id]
    except KeyError:
      self.subjects[cl.subject_id] = Subject(subject_id = cl.subject_id,
                                             p0 = self.config.p0,
                                             classes = self.config.label_map.keys())
#REMOVE RANDINT SECTION OF THE NEXT LINE - JUST INCLUDED FOR TESTING PURPOSES np.random.randint(0,10)!=0.
#'time_stamp' is the time at which the classification was processed by swap. 'classification_time' is the time recorded by Zooniverse as the time the classification was made.
    if cl.subject_id not in self.users[cl.user_id].user_subject_history:
        '''
        if from_csv:
            if (self.subjects[cl.subject_id].gold_label==-1) and \
               (self.subjects[cl.subject_id].score < self.config.thresholds[0]) and \
               (self.subjects[cl.subject_id].seen>=self.config.lower_retirement_limit):
                print('Beyond lower p threshold. Ignoring classification')
                return
            if (self.subjects[cl.subject_id].gold_label==-1) and \
               (self.subjects[cl.subject_id].seen >= self.config.retirement_limit):
                print('Beyond upper N_class threshold. Ignoring classification')
                return'''
        self.subjects[cl.subject_id].update_score(cl.label, self.users[cl.user_id],time_stamp,classification_time)
        if online:
          if self.subjects[cl.subject_id].gold_label==0:
#            print('Gold, dud, updating user score')
            self.users[cl.user_id].update_user_score(0, cl.label)
          if self.subjects[cl.subject_id].gold_label==1:
#            print('Gold, lens...')
            if self.subjects[cl.subject_id].hard_sim_label == 0 or (self.subjects[cl.subject_id].hard_sim_label == 1 and int(cl.label)==1):
#              print('... easy sim or (hard sim & correct), updating user score. Hard sim:' + str(self.subjects[cl.subject_id].hard_sim_label))
              self.users[cl.user_id].update_user_score(1, cl.label)
            else:
              print('hard sim, not updating user score')
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
            print("KEY_ERROR_8 HERE")
            with open(self.config.keyerror_list_path,'r') as g:
                keyerror_list=eval(g.read())
                keyerror_list[8]+=1
            with open(self.config.keyerror_list_path,'w') as q:
                q.write(str(keyerror_list))

#Takes as an input a list of subjects. If a subject is gold, it is not retired, otherwise if its score is past either threshold it is added to the to_retire list and marked as '0' or '1' depending on which threshold it passed.
  def retire(self, subject_batch):
    to_retire = []
    for subject_id in subject_batch:
      try:
        subject = self.subjects[subject_id]
      except KeyError:
        print('B: Subject {} is missing.'.format(subject_id))
        continue
      if subject.gold_label in (0,1):
        print('continuing as gold')
# if this is a gold standard image never retire
        continue
      subject.score=float(subject.score)
      if (subject.score < self.config.thresholds[0]) and subject.seen>=self.config.lower_retirement_limit:
        if subject.gold_label !=-1:
          print('Error: NOT CONTINUING!!'+str(type(subject.gold_label)))
        subject.retired = True
        self.subjects[subject_id].retired=True
#        print('setting retired_as to 0')
        subject.retired_as = 0
        to_retire.append(subject_id)
#        logging.info('retired as reached lower threshold')

      elif subject.score > self.config.thresholds[1] and subject.seen>=self.config.lower_retirement_limit:
        subject.retired = True
        self.subjects[subject_id].retired=True
        subject.retired_as = 1
        to_retire.append(subject_id)
#        logging.info('retired as reached upper threshold')
#MUST RETURN A LIST:
    return to_retire

#Takes as an input a list of subjects. If the subject is beyond the retirement limit (ie too many classifications), it is retired, with classification *based on the majority vote of its previous classifications, not a rounded version of its current score*.
  def retire_classification_count(self, subject_batch):
    def majority_vote(sequence):
      occurence_count = Counter(sequence)
      return occurence_count.most_common(1)[0][0]
    to_retire = []
    for subject_id in subject_batch:
      try:
        subject = self.subjects[subject_id]
      except KeyError:
        print('A: Subject {} is missing.'.format(subject_id))
        continue
      if subject.seen >= self.config.retirement_limit and subject.gold_label==-1:
        subject.retired = True
        self.subjects[subject_id].retired=True
        subject.retired_as = majority_vote([h[2] for h in subject.history[1:len(subject.history)]])
        logging.info('retired as reached retirement limit: '+ str(subject.retired_as))
        to_retire.append(subject_id)
#MUST RETURN A LIST:
    return to_retire

#Sends the subjects to be retired to panoptes.
  def send_panoptes(self, subject_batch,reason):
    subjects = []
    try:
        #with Pool(4) as pool:
        #    subjects = pool.map(find_subjects, subject_batch)
    #    for subject_id in subject_batch:
    #      subjects.append(PanoptesSubject().find(subject_id))
        #print('SUBJ',subjects)
        print('SUBJ BATCH',subject_batch)
        self.workflow.retire_subjects([str(s_i) for s_i in subject_batch],\
                                      reason=reason)#self.workflow.retire_subjects(subjects,reason=reason)
        for subject_i in subject_batch:
            print(subject_i,self.workflow.subject_workflow_status([subject_i]).retirement_reason)
    except Exception as excep:
        print(excep)
        print('KEY_ERROR_4 HERE')
        with open(self.config.keyerror_list_path,'r') as g:
            keyerror_list=eval(g.read())
            keyerror_list[4]+=1
        with open(self.config.keyerror_list_path,'w') as q:
            q.write(str(keyerror_list))

  def retrieve_list(self,list_path):
    import pandas as pd
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
    import pandas as pd
    list_db = pd.DataFrame({'Column 1': list_total}) #Dont change this column title unless change those in def retrieve_list() too.
    list_db.to_csv(list_path, index=False)

  #Goes through each row in the csv, defining classification events, updating scores (as per process_classification) and labelling subjects to be retired if required.
  def process_classifications_from_csv_dump(self, path, online=False):
    with open(path, 'r',encoding='utf-8-sig') as csvdump:
      reader = csv.DictReader(csvdump)
      for row in reader:
        id = int(row['classification_id'])
        classification_time=row['created_at']
        try:
          assert int(row['workflow_id']) == self.config.workflow
        except AssertionError as e:
           print(int(row['workflow_id']),'Assertion Error in proccess_classifications_from_csv_dump: Ignoring this classification')
           continue
        try:
          user_id = int(row['user_id'])
        #IGNORING NOT-LOGGED-ON USERS (and 5 lines below too):
          #if str(row['user_id'][0:13])=='not-logged-in':
          #  continue
        except ValueError:
          user_id = row['user_name']
        #IGNORING NOT-LOGGED-ON USERS (and 5 lines above too):
          #if str(row['user_name'][0:13])=='not-logged-in':
          #  continue
        dt_row = row['created_at']
        subject_id = int(row['subject_ids'])
        annotation = json.loads(row['annotations'])
        ###NOW IGNORING THESE FLAGS:
        try:
          already_seen_i=json.loads(row['metadata'])['subject_selection_state']['already_seen']
          already_seen_i_2=json.loads(row['metadata'])['seen_before']
          self.instance_counts[0]+=1
          if already_seen_i!=already_seen_i_2:
            print('Difference here: '+ str(already_seen_i)+" "+str(already_seen_i_2))
            already_seen_i=True
            self.instance_counts[3]+=1
        ###NOW IGNORING THESE FLAGS:
        except KeyError:
          try:
            already_seen_i=json.loads(row['metadata'])['subject_selection_state']['already_seen']
            self.instance_counts[1]+=1
          except KeyError:
            try:
                already_seen_i=json.loads(row['metadata'])['seen_before']
                self.instance_counts[4]+=1
            except:
                already_seen_i=False
                self.instance_counts[2]+=1
        ###IGNORING ANY ALREADY_SEEN FLAGS:
        #if already_seen_i==False:
        if True:
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
        else:
          print('duplicate classification' + str((subject_id,user_id)))
          self.instance_counts[5]+=1
#OCT 2022: IGNORING ALL OF THIS: HAVE ALREADY GOT A FUNCTION WITHIN PROCESS_CLASSIFICATIONS WHICH CHECKS THAT THE SUBJECT IS NOT ALREADY IN THE USER'S HISTORY - CAN USE THIS ON ITS OWN TO DO THE SAME THING AND THUS IGNORE THE COMPLICATIONS OF THE already_seen AND seen_before FLAGS.
#Note, sometimes a user has already seen a subject before in their history (as it is caught by the function within process_classifications), but doesn't have either already-seen/seen-before flags. From testing, this occurs for ~5000 of 46,000 subjects which users classified more than once.
      print("On testing, it was found a classification csv may contain both 'already_seen' and 'seen_before' flags. Out of 1873 classifications, 1286 had both flags, 585 only had 'already_seen' and 2 only had 'seen_before'. In 10 instances, the flags differed (ie one said True, the other False). For some reason, all the 'seen_before' flags were True. It seems this 'seen_before' flag is historical. In the few instances where there is no 'already_seen' flag, it is assumed that the user has *not* seen the subject before (OCT 2022: CHANGED AS ABOVE), however in cases where the flags differ, it is assumed the user *has* seen the subject before (to be on the safe side, and why else would the seen_before flag be present?). self.instance_counts gives: [0]: number of instances where both flags are present, [1]: number of instances where only 'already_seen' flag present, [2]: number of instances where neither 'already_seen' nor 'seen_before' flag is not present (ie number of instances the already_seen flag is decided a priori to be False), [3]: number of instances where the values for 'already_seen' and 'seen_before' differ (when both present), in which case already_seen is assumed to be True, [4]: number of instances where 'seen before' was present, but 'already_seen' was not,[5]: total number of times already_seen_i=True and thus the classification was ignored")
      print("To revert to Fridays graph, remove already_seen_i=True from case where two flags differ")
      print("For plotting Dec graph, have changed 1) config file db name, 2) config file gold file paths 3) run.py gold file paths 4) run.py test_caesar/online call function 5) kswap_plots database path")
      print('Instance Counts: [Both, AS only, !AS and !SB, Dif,SB only,Tot ign]' + str(self.instance_counts))
#***Why is there an try/except function here? Surely need to do both simultaneously?***
#Original:
#    try:
#      self.retire(self.subjects.keys())
#    except TypeError:
#      self.retire_classification_count(self.subjects.keys())
###Suggested update:
#    retired_list=retrieve_list(self.config.retired_items_path)
#    retired_list_threshold_updates=self.retire(self.subjects.keys())
#    retired_list_Nclass_updates=self.retire_classification_count(self.subjects.keys())
#    self.send_panoptes(retired_list_threshold_updates,'consensus')
#    self.send_panoptes(retired_list_Nclass_updates,'classification count')
#    retired_list.extend(list(set(np.array(retired_list_threshold_updates+retired_list_Nclass_updates))))
#    self.save_list(retired_list,self.config.retired_items_path)

  def classify_rows(self,row_list,online):
          random.shuffle(row_list)
          for r in range(len(row_list)):
            row_i = row_list[r]
            id = int(row_i['classification_id'])
            #assert int(row_i['workflow_id']) == self.config.workflow
            try:
              user_id = int(row_i['user_id'])
            except ValueError:
              user_id = row_i['user_name']
            subject_id = int(row_i['subject_ids'])
            annotation = json.loads(row_i['annotations'])
            #already_seen_i=json.loads(row['metadata'])['subject_selection_state']['already_seen']
            classification_time=row_i['created_at']
            #if already_seen_i==False:
            cl = Classification(id,
                  user_id,
                  subject_id,
                  annotation,
                  label_map=self.config.label_map,
                  classification_time=classification_time)
            self.process_classification(cl, online)
            #else:
            #  print('duplicate classification' + str((subject_id,user_id)))

#Returns a timestamp for the time the classification was made
  def date_time_convert(self, row_created_at):
      s_i=[]
      for s in range(len(row_created_at)):
        if row_created_at[s]=='-' or row_created_at[s]==':':
          s_i.append(s)
      dt_i = datetime(int(row_created_at[0:s_i[0]]), int(row_created_at[s_i[0]+1:s_i[1]]), int(row_created_at[s_i[1]+1:s_i[1]+3]),int(row_created_at[s_i[2]-2:s_i[2]]), int(row_created_at[s_i[2]+1:s_i[3]])).timestamp()
      return dt_i

  def process_classifications_from_shuffled_csv(self, path, shuffle_time,online=False):
    with open(path, 'r') as csvdump:
      reader = csv.DictReader(csvdump)
      q='initialise'
      row_list=[]
      i=0;f = 0
      for row in tqdm(reader):
        #if f<shuffle_time:
        #    row_list.append(row)
        #    f+=1
        #else:
        #    self.classify_rows(row_list,online)
        #    f=0
        #    row_list=[row]
          if i==0:
            dt_0 = self.date_time_convert(row['created_at'])
            row_list.append(row)
            i+=1
          else:
            dt_i = self.date_time_convert(row['created_at'])
            if dt_i<dt_0+shuffle_time:
              row_list.append(row)
            else:
              self.classify_rows(row_list,online)
              row_list=[row]
              dt_0 = self.date_time_convert(row['created_at'])
      self.classify_rows(row_list,online)
#Recieves data from caesar, adds classification events/updating scores (as per process_classification), but doesnt set any to be retired.
  def caesar_recieve(self, ce):
    import logging
    import time
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    keyerror_1_i=0;keyerror_2_i=0
    try:
#        data = ce.Extractor.next()
        st = time.time()
#Located here:
#./opt/anaconda3/lib/python3.8/site-packages/caesar_external/utils/caesar_utils.py:        return cls.instance().sqs_retrieve_fast(queue_url)
#./opt/anaconda3/lib/python3.8/site-packages/caesar_external/extractor.py:    def sqs_retrieve_fast(self):
        data = ce.SQSExtractor.sqs_retrieve_fast()
        haveItems = False
        subject_batch = []
        q=0;q_all=0
#        for i, item in enumerate(data):
        print(f'Beginning processing of {len(data)} subjects')
        for i in range(len(data)):
          item = data[i]
          try:
              q_all+=1
              haveItems = True
              id = int(item['id'])
              already_seen_i=item['already_seen']
              try:
                user_id = int(item['user'])
              except ValueError:
                user_id = item['user']
              except TypeError as e:
                print(e)
                print(item)
                print('No user identification')
                continue
#              print('')
#              print('User id: ' + str(user_id))
              subject_id = int(item['subject'])
              #subject_i = self.subjects[subject_id]
              classification_time=item['classification_time']
#              print('Gold status: ' + str(subject_i.gold_label))
#              print('Previous Panoptes Retirement Reason: '+ str(self.workflow.subject_workflow_status(subject_id).retirement_reason))
#              print('Previous Database Status: ' + str(bool(subject_i.retired)))
#              if subject_i.retired==True:
#                print('Previous Database Retirement Classification: ' + str(subject_i.retired_as)+ ' vs gl: ' + str(subject_i.gold_label))
#              else:
#                print('Previous Database Retirement Classification: ' + str(subject_i.retired_as))
#              print('Subject Score: '+str(subject_i.score))
#              print('')
              try:
                assert self.workflow.subject_workflow_status([subject_id]).retirement_reason == None
              except:
                 pass
#                print('Subject has already been retired?! ' + str(subject_id))
              annotation = item['annotations']
#              if already_seen_i==True:
        #Note 'already seen' could now mean the flag wasn't present in AWS and was hence ignored.
#                print(str(user_id) + ' has already seen this subject: '+ str(subject_id))
        #CHECK THIS IF STATEMENT BELOW IS CORRECT BEFORE RUNNING:
#              if already_seen_i==False:
              if True:
                  q+=1
                  print('Making classification')
                  cl = Classification(id, user_id, subject_id, annotation,label_map=self.config.label_map,classification_time=classification_time)
                  print('Processing classification')
                  self.process_classification(cl,online=True)
                  #^Adds users to User, subjects to Subjects and updates scores/history's accordingly.
                  self.last_id = id
                  self.seen.add(id)
                  subject_batch.append(subject_id)
              else:
                print('SUBJECT IGNORED AS REQUIRED')
          except KeyError as e:
            print('KEYERROR_1 HERE')
            print(e)
            keyerror_1_i+=1
            continue
        et=time.time()
        if q!=q_all:
            logging.info(str(q_all-q) + ' already_seen subjects ignored')
        if q!=0:
            logging.info('time taken = ' + str(et-st)+ ' for ' + str(q) + ' classifications, ' + str((et-st)/q) + 's/cl')
        else:
            logging.info('time taken = ' + str(et-st)+ ' for ' + str(q) + ' classifications.')
        return haveItems, subject_batch,q,q_all,keyerror_1_i,keyerror_2_i, q_all
    except Exception as ex:
        print('KEYERROR_2 HERE')
        print(ex)
        keyerror_2_i+=1
        return False, [], 0,0,keyerror_1_i,keyerror_2_i,0
    
  def process_classifications_from_caesar(self, caesar_config_name):
    ce.Config.load(caesar_config_name)
    with open(self.config.aws_list_path,'r') as f: #('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/AWS_list.txt', 'r') as f:
        aws_list=eval(f.read())
    q=0
    k=0
    print('README:')
    print('Notes: 1) Need to rename/refresh swap.db database when run beta test')
    print('       2) Need to make sure the subjects are all unretired before starting (they are all at the bottom of this file in the subject_set list)')
    print('       3) Need to email cliff again about the delay time before images are no longer offered for classification on zooniverse')
    print('       4) Need to make sure _already_seen_ IF statement is correct above, ie it isnt just _if True:_ as otherwise duplicate classifications can happen')
    print('       5) Adjust config file to change retirement thresholds before running, and can change file paths here as needed ')
    save_timer_start = time.time()
    try:
      while True:
        ST_time = time.time()
        haveItems, subject_batch,q,q_all,keyerror_1_i,keyerror_2_i,N_proc = self.caesar_recieve(ce)
#        process_time_array_x[N_proc]+=1
#        process_time_array_y[N_proc].append(time_proc)
#aws_list[q] = Number of times a retrieval has been made containing q messages.
        try:
          aws_list[q]+=1
        except:
          pass
        if haveItems:
          k=0
          print(aws_list)
          st_save=time.time()
#          self.save()
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
        logging.info('Retiring ' + str(len(retire_batch))+' subjects: ' +\
                                   str(retire_batch))
        try:
            retired_list = self.retrieve_list(self.config.retired_items_path)
            retired_list.extend(list(set(np.array(retire_list_thres+retire_list_Nclass))))
            self.save_list(retired_list,self.config.retired_items_path)
        except Exception as ex:
            print(ex)
            with open(self.config.keyerror_list_path, 'r') as g:
                keyerror_list=eval(g.read())
            print('KEY_ERROR_6 HERE')
            keyerror_list[6]+=1
            with open(self.config.keyerror_list_path, 'w') as h:
                h.write(str(keyerror_list))
        if len(retire_batch)>0:
            try:
                if len(retire_list_Nclass)>0:
                    retired_list = self.retrieve_list(self.config.Nclass_retirement_path)
                    retired_list.append([time.time(),retire_list_Nclass])
                    self.save_list(retired_list,self.config.Nclass_retirement_path)
            except:
                print("Couldn't append subjects for retirement to file,1")
                pass
            try:
                if len(retire_list_thres)>0:
                    retired_list = self.retrieve_list(self.config.Nthres_retirement_path)
                    retired_list.append([time.time(),retire_list_thres])
                    self.save_list(retired_list,self.config.Nthres_retirement_path)
            except:
                print("Couldn't append subjects for retirement to file,2")
                pass
#GET RID OF THIS UNRETIREMENT LINE!!! JUST NEEDED FOR TESTING TEMPORARILY
#        self.workflow.unretire_subjects(np.array(retire_batch).tolist())
#        for f in range(len(retire_batch)):
#          if self.workflow.subject_workflow_status([retire_batch[f]]).retirement_reason!=None:
#            print('RETIREMENT ERROR HERE')
        ST_retirement=time.time()
#        self.send_panoptes(retire_list_thres,'consensus')
#        self.send_panoptes(retire_list_Nclass,'classification_count')
        print('Sending to panoptes')
        self.send_panoptes(retire_batch,'consensus')
        print('Sent to panoptes')
        st_save2 = time.time()
#        self.save()
        et_save2 = time.time()
        print('Single save time: ' + str(et_save2-st_save2))
        ED_retirement=time.time()
        logging.info('Retirement time: ' + str((ED_retirement-ST_retirement)/np.max([1,len(retire_batch)])) + '/subj for ' +  str(len(retire_batch)) + ' retirements')
#        for f in range(len(retire_batch)):
#          if self.workflow.subject_workflow_status([retire_batch[f]]).retirement_reason!='consensus':
#            print('RETIREMENT ERROR 2 HERE')
#        #Number retired may include double counting as subjects can be retired by both consensus or classification_count, but use this as want total number of subjects sent for retirement.
#        number_retired = int(len(set(np.array(retire_list_thres)))+len(set(np.array(retire_list_Nclass))))
#KEEP NEXT LINE IN! SAVES DATABASE EVERY 30MINS (now 10s, for testing):
        if time.time()-save_timer_start>300:#1800:
          print('Starting saving')
          self.save()
          save_timer_start = time.time()
          print('Finished saving')
        if keyerror_1_i != 0 or keyerror_2_i !=0:
            with open(self.config.keyerror_list_path,'r') as g: #('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/KeyError_list.txt', 'r') as g:
              keyerror_list=eval(g.read())
            keyerror_list[0]+=keyerror_1_i
            keyerror_list[1]+=keyerror_2_i
            with open(self.config.keyerror_list_path,'w') as q:#('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/KeyError_list.txt', 'w') as q:
              q.write(str(keyerror_list))
        with open(self.config.aws_list_path,'w') as f:#('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/AWS_list.txt', 'w') as f:
          f.write(str(aws_list))
        print('')
        ED_time = time.time()
        print('Total loop time per subject:' + str((ED_time-ST_time)/np.max([N_proc,1])) +' for ' + str(N_proc) + ' classifications')
    except KeyboardInterrupt as e:
#New save func:
      print('Starting save')
      self.save()
      print('Finishing save')
#      print(retirement_time_array_x,[np.median(retirement_time_array_y[i]) for i in range(len(retirement_time_array_y))],[np.mean(retirement_time_array_y[i]) for i in range(len(retirement_time_array_y))])
#      st_1 = time.time()
#      self.send_panoptes(list(set(np.array(test_retirement_list))),'classification_count')
#      print('Retirement time: ' + str(time.time()-st_1))
#      print('... for ' + str(len(list(set(np.array(test_retirement_list))))) + 'subjects')
###Retirements:
      retire_list_thres=self.retire(subject_batch)
      retire_list_Nclass=self.retire_classification_count(subject_batch)
      retire_batch = list(set(np.array(retire_list_thres+retire_list_Nclass)))
      logging.info('Retiring ' + str(len(retire_batch))+ ' subjects: ' + str(retire_batch))
#      self.send_panoptes(retire_list_thres,'consensus')
#      self.send_panoptes(retire_list_Nclass,'classification_count')
      self.send_panoptes(retire_batch,'consensus')
      try:
        with open(self.config.aws_list_path,'w') as f:#('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/AWS_list.txt', 'w') as f:
          f.write(str(aws_list))
      except Exception as exep:
        print(exep)
        print('cant save AWS list for final iteration')
        pass
      print('Received KeyboardInterrupt {}'.format(e))
      self.save()
      print('Terminating SWAP instance.')
      exit()
    except Exception as exep1:
        print('KEY_ERROR_7 HERE')
        print(exep1)
        with open(self.config.keyerror_list_path, 'r') as g:
            keyerror_list=eval(g.read())
        keyerror_list[7]+=1
        with open(self.config.keyerror_list_path, 'w') as h:
            h.write(str(keyerror_list)) 
        self.save()
###Retirements:
    st=time.time()
    logging.info('Retiring ' + str(len(set(np.array(retire_list_thres+retire_list_Nclass))))+ ' subjects')
    logging.info('To retire: ' + str(set(np.array(retire_list_thres+retire_list_Nclass))))
#    self.send_panoptes(retire_list_thres,'consensus')
#    self.send_panoptes(retire_list_Nclass,'classification_count')
    self.send_panoptes(list(set(np.array(retire_list_thres+retire_list_Nclass))),'consensus')
    self.save()
    with open(self.config.aws_list_path,'w') as f: #('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/AWS_list.txt', 'w') as f:
      f.write(str(aws_list))
    et=time.time()
    logging.info('retirement time: ' + str(et-st))

#Adds gold subjects from the csv to Subject class.
  def get_golds(self, path, hard_sims_path = None):
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

#Goes through the csv file, if subject is a gold, updates the user-score accordingly.
#NOTE (April 2023) I'm not sure there is anything in this bit of code which prevents (repeatedly) updating user scores if they see the same gold subjects repeatedly? This only matters for run_offline though.
  def apply_golds(self, path):
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
#          # ignore repeat classifications of the same subject
        except AssertionError as e:
          print('Assertion Error in apply_golds')
                  
        try:
          cl = Classification(id,
                              user_id,
                              subject_id,
                              annotation,
                              label_map=self.config.label_map,
                              classification_time=classification_time)
        except ValueError as e2:
          print(e2)
          continue
        try:
          self.users[cl.user_id]
        except KeyError:
          self.users[cl.user_id] = User(user_id = cl.user_id,
                                        classes = self.config.label_map.keys(),
                                        gamma   = self.config.gamma,
                                        user_default = self.config.user_default)
        
        try:
          gold_label = self.subjects[cl.subject_id].gold_label
          if gold_label==0:
            self.users[cl.user_id].update_user_score(gold_label, cl.label)
          elif gold_label==1 and self.subjects[cl.subject_id].hard_sim_label==0:
            self.users[cl.user_id].update_user_score(gold_label, cl.label)
          elif gold_label==1 and self.subjects[cl.subject_id].hard_sim_label==1 and cl.label == gold_label:
            self.users[cl.user_id].update_user_score(gold_label, cl.label)
        except KeyError as e:
          print('KEY ERROR IN APPLY GOLDS.')
          continue

###Code to Mass-Unretire subjects from a subject-workflow csv.
  def unretire_all_subjects_in_csv(self,csv_file):
    subject_id_set=[]
    with open(csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for i,row in enumerate(reader):
            subject_id_set.append(row['subject_id'])
    print('Unretiring')
    self.workflow.unretire_subjects(subject_id_set)
    print('Unretired')
    for i in range(len(subject_id_set)):
        print(subject_id_set[i])
        print(self.workflow.subject_workflow_status([subject_id_set[i]]).retirement_reason)

#Run offline - NOTE: See April 2023 comment above apply_golds.
  def run_offline(self, gold_csv, classification_csv, hard_sims_csv=None):
    self.get_golds(gold_csv, hard_sims_csv)
    self.apply_golds(classification_csv)
    self.process_classifications_from_csv_dump(classification_csv)

#Run Online (misses out on apply_golds but user-scores are updated within process_classification function for online mode.)
  def run_online(self, gold_csv, classification_csv,hard_sims_csv=None):
    self.get_golds(gold_csv,hard_sims_csv)
    self.process_classifications_from_csv_dump(classification_csv, online=True)
    print('Total ignored:     ',total_ignored)
    print('Total not ignored: ',total_not_ignored)

  def run_shuffled_online(self, gold_csv, classification_csv,shuffle_time,hard_sims_csv=None):
    st_shuffle = time.time()
    self.get_golds(gold_csv,hard_sims_csv)
    self.process_classifications_from_shuffled_csv(classification_csv,shuffle_time, online=True)
    print('Time Taken: ',time.time()-st_shuffle)
    print('Total ignored:     ',total_ignored)
    print('Total not ignored: ',total_not_ignored)

  def run_caesar(self,gold_csv,hard_sims_csv=None):
#    print('Unretiring all subjects')
#    self.unretire_all_subjects_in_csv(gold_csv)
#    print('Finished unretiring all subjects')
    self.load()
    self.get_golds(gold_csv,hard_sims_csv)
    self.process_classifications_from_caesar('test_config2')
#    a = list((self.retrieve_list(self.config.retired_items_path).copy()))
#    a_set=list(set(self.retrieve_list(self.config.retired_items_path).copy()))
#    N_retired = 0
#    N_active=0
#    for i in range(len(a_set)):
#      if self.workflow.subject_workflow_status(a_set[i]).retirement_reason == None:
#        N_active+=1
#      if self.workflow.subject_workflow_status(a_set[i]).retirement_reason != None:
#        N_retired+=1
#    if len(a_set)==len(a):
#        print('N_retired = ' + str(N_retired) + ', N_active = ' + str(N_active))
#    else:
#        print('ERROR HERE: some subjects not retired normally')
#        print('N_retired = ' + str(N_retired) + ', N_active = ' + str(N_active))

