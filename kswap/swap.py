import os
import csv
import json
import sqlite3
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
from config import Config

from collections import Counter

from panoptes_client import Panoptes, Workflow
from panoptes_client import Subject as PanoptesSubject
from panoptes_client.panoptes import PanoptesAPIException

try:
    import caesar_external as ce
except ModuleNotFoundError:
    pass

#**See note line 459**
#Classification Class:
#Arguments: Classification_id, User_id, Subject_id, Annotation, Label_map
#For the time-being, if annotations is not '[]', classifies as a lens, if not, classifies as not-a-lens.
class Classification(object):
  def __init__(self,
               id,
               user_id,
               subject_id,
               annotation,
               label_map):
      
    self.id = int(id)
    try:
      self.user_id = int(user_id)
    except ValueError:
      self.user_id = user_id
    self.subject_id = int(subject_id)
    self.label_map  = label_map
    self.label = self.parse(annotation)

  def parse(self, annotation):
    value = annotation['T1'][0]['value']
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
    self.history = [('_', self.user_score)]
  
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
            json.dumps(self.history))

#Subject Class:
#Arguments: Subject_ID, p0 (ie initial probability), classes (0,1) , gold-label (default= -1 ie not test), k (ie N. classes), retired (=False).
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
               epsilon=1e-9):
      
    self.subject_id = subject_id
    self.score = float(p0)
    self.classes = classes
    self.k = len(self.classes)
    self.gold_label = gold_label
    self.epsilon = epsilon
    self.retired = False
    self.retired_as = None
    self.seen = 0
    self.history = [('_', '_', '_', self.score)]

  def update_score(self, label, user):
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
    self.history.append((user.user_id, user.user_score, label, self.score))
    self.seen += 1

  def dump(self):
    return (self.subject_id, \
            self.gold_label, \
            json.dumps(self.score), \
            self.retired, \
            self.retired_as, \
            self.seen, \
            json.dumps(self.history))

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
#DELETE THE NEXT LINE - JUST THERE TO CHECK RETIREMENTS ARE WORKING. IN THIS LINE, ID=WORKFLOW ID
    self.id=self.config.workflow
    self.seen = set([])
    self.db_exists = False
    self.timeout = timeout # wait x seconds to acquire db connection
    try:
      self.create_db()
      self.save()
    except sqlite3.OperationalError:
      self.db_exists = True

    Panoptes.connect(username='###', \
                     password='###')
                     
    self.workflow = Workflow.find(config.workflow)
    
  def connect_db(self):
    return sqlite3.connect(self.config.db_path+self.config.db_name, timeout=self.timeout)

  def create_db(self):
    conn = self.connect_db()
    conn.execute('CREATE TABLE users (user_id PRIMARY KEY, user_score, ' +\
                 'confusion_matrix, history)')

    conn.execute('CREATE TABLE subjects (subject_id PRIMARY KEY, ' +\
                 'gold_label, score, retired, retired_as, seen ,history)')
                 
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
      self.subjects[subject['subject_id']].seen = subject['seen']
      self.subjects[subject['subject_id']].history = json.loads(subject['history'])

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

    c.executemany('INSERT OR REPLACE INTO users VALUES (?,?,?,?)',
                   self.dump_users())

    c.executemany('INSERT OR REPLACE INTO subjects VALUES (?,?,?,?,?,?,?)',
                   self.dump_subjects())
                   
    c.execute('INSERT OR REPLACE INTO config VALUES (?,?,?,?,?,?,?,?,?,?,?)',
               self.dump_config())

    conn.commit()
    conn.close()
#Checks if classification is new (if not, function does nothing), if the user is new (if so, makes them a member of User class), if subject is new (if so, makes them a member of Subject class).
#Then: Updates subject score due to this classification. If subject is a gold and swap being run in online mode, it also updates user score and adds it to user history.
  def process_classification(self, cl, online=False):
    # check if classification already seen
#ALSO HASHING OUT THIS - MAY NEED TO UNHASH LATER:
#    if cl.id <= self.last_id:
#      return
    # check user is known
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
    logging.info('Adding to Subject list: ' + str(cl.subject_id))

    self.subjects[cl.subject_id].update_score(cl.label, self.users[cl.user_id])
    if self.subjects[cl.subject_id].gold_label in (0,1) and online:
      gold_label = self.subjects[cl.subject_id].gold_label
      self.users[cl.user_id].update_user_score(gold_label, cl.label)
    self.users[cl.user_id].history.append((cl.subject_id, self.users[cl.user_id].user_score))
    self.last_id = cl.id
    self.seen.add(cl.id)

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
# if this is a gold standard image never retire
        continue
      subject.score=float(subject.score)
      if subject.score < self.config.thresholds[0]:
        subject.retired = True
        subject.retired_as = 0
        to_retire.append(subject_id)
        logging.info('retired as reached lower threshold')

      elif subject.score > self.config.thresholds[1]:
        subject.retired = True
        subject.retired_as = 1
        to_retire.append(subject_id)
        logging.info('retired as reached upper threshold')

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
        subject.retired_as = majority_vote([h[2] for h in subject.history])
        logging.info('retired as reached retirement limit')
        to_retire.append(subject_id)
    return to_retire

#Sends the subjects to be retired to panoptes.
  def send_panoptes(self, subject_batch,reason):
    subjects = []
    for subject_id in subject_batch:
      subjects.append(PanoptesSubject().find(subject_id))
    self.workflow.retire_subjects(subjects,reason=reason)

  def make_tuples_list(self):
    import pandas as pd
    tuples_path = self.config.tuples_path
    try:
        df = pd.read_csv(tuples_path)
        tuples = []
        for row in df.iterrows():
            tuples_i = eval(row[1]['tuples'])
            tuples.append(tuples_i)
        return tuples
    except FileNotFoundError:
        return []

  def save_tuples_list(self,tuples_list):
    import pandas as pd
    tuples_path = self.config.tuples_path
    tuples_db = pd.DataFrame({'tuples': tuples_list})
    tuples_db.to_csv(tuples_path, index=False)

#Goes through each row in the csv, defining classification events, updating scores (as per process_classification) and labelling subjects to be retired if required.
  def process_classifications_from_csv_dump(self, path, online=False):
    tuples_list = self.make_tuples_list()
    with open(path, 'r') as csvdump:
      reader = csv.DictReader(csvdump)
      for row in reader:
        id = int(row['classification_id'])
        try:
          assert int(row['workflow_id']) == self.config.workflow
#          # ignore repeat classifications of the same subject
#          json.loads(row['metadata'])['seen_before']
#          continue
#        except KeyError as e:
##          pass
        except AssertionError as e:
          print('Assertion Error in proccess_classifications_from_csv_dump')
#          print(e, row)
#          continue
        try:
          user_id = int(row['user_id'])
        except ValueError:
          user_id = row['user_name']
        subject_id = int(row['subject_ids'])
        annotation = json.loads(row['annotations'])
        if (subject_id,user_id) not in tuples_list:
          tuples_list.append((subject_id,user_id))
          try:
              cl = Classification(id,
                                  user_id,
                                  subject_id,
                                  annotation,
                                  label_map=self.config.label_map)
          except ValueError:
            continue
          self.process_classification(cl, online)
    self.save_tuples_list(tuples_list)
    try:
      self.retire(self.subjects.keys())
    except TypeError:
      self.retire_classification_count(self.subjects.keys())

#Recieves data from caesar, adds classification events/updating scores (as per process_classification), but doesnt set any to be retired.
  def caesar_recieve(self, ce):
    import logging
    tuples_list = self.make_tuples_list()
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    data = ce.Extractor.next()
    haveItems = False
    subject_batch = []
    q=0
    for i, item in enumerate(data):
      q+=1
      haveItems = True
      id = int(item['id'])
#      try:
#        # ignore repeat classifications of the same subject
#      if item['already_seen']:
#        logging.info('continuing as already seen')
#        continue
#      except KeyError as e:
#        logging.info('key_error here')
#        continue
#PROBABLY NEED TO UNHASH THIS AT SOME POINT:
#      if id < self.last_id:
#        logging.info('continuing here')
#        continue

      try:
        user_id = int(item['user'])
      except ValueError:
        user_id = item['user']
      subject_id = int(item['subject'])
#      object_id = item['object_id']
#NEED TO CHECK THESE ANNOTATIONS ARE THE SAME TYPE AS ABOVE
      annotation = item['annotations']
      if (subject_id,user_id) not in tuples_list:
          tuples_list.append((subject_id,user_id))
          cl = Classification(id, user_id, subject_id, annotation,label_map=self.config.label_map)
          self.process_classification(cl,online=True)
          #^Adds users to User, subjects to Subjects and updates scores/history's accordingly.
          self.last_id = id
          self.seen.add(id)
          subject_batch.append(subject_id)
    self.save_tuples_list(tuples_list)
    return haveItems, subject_batch

  def process_classifications_from_caesar(self, caesar_config_name):
    import logging
    import numpy as np
    ce.Config.load(caesar_config_name)
    q=0
    retire_list_thres=[]
    retire_list_Nclass=[]
    try:
      while True:
        haveItems, subject_batch = self.caesar_recieve(ce)
        if haveItems:
          self.save()
          ce.Config.instance().save()
          # load the just saved ce config
          ce.Config.load(caesar_config_name)
#          self.send(ce, subject_batch)
        print('breaking')
        retire_list_thres.extend(self.retire(subject_batch))
        retire_list_Nclass.extend(self.retire_classification_count(subject_batch))
        logging.info('To retire' + str((np.array(retire_list_thres+retire_list_Nclass))))
    except KeyboardInterrupt as e:
      retire_list_thres.extend(self.retire(subject_batch))
      retire_list_Nclass.extend(self.retire_classification_count(subject_batch))
      self.send_panoptes(retire_list_Nclass,'human')
#      self.send_panoptes(retire_list_thres,'consensus')
#      self.send_panoptes(retire_list_Nclass,'classification_count')
      logging.info('Retiring ' + str(len(np.array(retire_list_thres+retire_list_Nclass)))+ ' subjects')
      print('Received KeyboardInterrupt {}'.format(e))
      self.save()
      print('Terminating SWAP instance.')
      exit()
#    send_panoptes(retire_list_thres,'consensus')
#    send_panoptes(retire_list_Nclass,'classification_count')

#Adds gold subjects from the csv to Subject class.
  def get_golds(self, path):
    with open(path,'r') as csvdump:
      reader = csv.DictReader(csvdump)
      for row in reader:
        subject_id = int(row['subject_id'])
        gold_label = int(row['gold'])
        self.subjects[subject_id] = Subject(subject_id,
                                            classes = self.config.label_map.keys(),
                                            p0 = self.config.p0,
                                            gold_label = gold_label)

#Goes through the csv file, if subject is a gold, updates the user-score accordingly.
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
        try:
          assert int(row['workflow_id']) == self.config.workflow
#          # ignore repeat classifications of the same subject
#          if json.loads(row['metadata'])['seen_before']:
#            continue
#        except KeyError as e:
#          print(e, row)
#          pass
        except AssertionError as e:
          print('Assertion Error in apply_golds')
#          print(e, row)
#          continue
                  
        try:
          cl = Classification(id,
                              user_id,
                              subject_id,
                              annotation,
                              label_map=self.config.label_map)
        except ValueError:
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
          assert gold_label in (0 ,1)
          self.users[cl.user_id].update_user_score(gold_label, cl.label)
        except AssertionError as e:
          continue
        except KeyError as e:
          continue

#Run offline
  def run_offline(self, gold_csv, classification_csv):
    self.get_golds(gold_csv)
    self.apply_golds(classification_csv)
    self.process_classifications_from_csv_dump(classification_csv)

#Run Online (misses out on apply_golds but user-scores are updated within process_classification function for online mode.)
  def run_online(self, gold_csv, classification_csv):
    self.get_golds(gold_csv)
    self.process_classifications_from_csv_dump(classification_csv, online=True)

  def run_caesar(self,gold_csv):
    self.get_golds(gold_csv)
    self.process_classifications_from_caesar('test_config2')
#     print(self.workflow.subject_workflow_status(37775519).retirement_reason)
#     self.send_panoptes([37775519],'Test Retirement')
#     print(self.workflow.subject_workflow_status(37775519).retirement_reason)
#     self.workflow.unretire_subjects(37775586)
#     print(self.workflow.subject_workflow_status(37775586).retirement_reason)
#     self.send_panoptes([37775586],'human')
#     a = [37775637, 37775636, 37775636, 37775633, 37775628, 37775623, 37775620, 37775629]
#     for i in range(len(a)):
#        print(self.workflow.subject_workflow_status(a[i]).retirement_reason)
#        self.workflow.unretire_subjects(a[i])
