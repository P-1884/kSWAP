

"""
Get classifications from Panoptes or SQS Queue
"""

from caesar_external.utils.caesar_utils import Client, SQSClient,UniqueMessage
from caesar_external.data import Config
import logging
import time
logger = logging.getLogger(__name__)
import traceback
import sys
#    # or
#    print(sys.exc_info()[2])
import boto3
import os
import hashlib
import numpy as np
import json
from config import Config as config_swap
from pandas import DataFrame as DF
import pandas as pd
from multiprocessing import Pool

def retrieve_list(list_path):
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

def save_list(list_total,list_path):
#    print('About to save',list_total)
    list_db = pd.DataFrame({'Column 1': list_total}) #Dont change this column title unless change those in def retrieve_list() too.
    list_db.to_csv(list_path, index=False)

class Extractor:

    @classmethod
    def last_id(cls, next_id=None):
        if next_id is None:
          return Config.instance().last_id
        Config.instance().last_id = next_id

    @classmethod
    def next(cls):
        cl = None
        for cl in cls.get_classifications(cls.last_id()):
            yield cl
        if cl:
            cls.last_id(cl['id'])

    @classmethod
    def get_classifications(cls, last_id=None):
        logger.debug('Getting classifications\n{}\n{}'.format(
            last_id, Config.instance().sqs_queue))
        if Config.instance().sqs_queue is not None:
            return SQSExtractor.get_classifications(Config.instance().sqs_queue)
        return StandardExtractor.get_classifications(last_id)

    @classmethod
    def next_fast(cls):
      return SQSExtractor.get_classifications_fast(Config.instance().sqs_queue)

class StandardExtractor(Extractor):

    @classmethod
    def get_classifications(cls, last_id):
        logger.debug('Getting classifications from Panoptes')
        project = Config.instance().project
        for c in Client.extract(project, last_id):
            cl = {
                'id': int(c.id),
                'subject': int(c.links.subjects[0].id),
                'project': int(c.links.project.id),
                'workflow': int(c.raw['links']['workflow']),
                'annotations': c.annotations,
            }

            if cl['workflow'] != Config.instance().workflow:
                continue

            if 'user' in c.raw['links']:
                cl.update({'user': c.raw['links']['user']})
            else:
                session = c.raw['metadata']['session'][:10]
                cl.update({'user': 'not-logged-in-%s' % session})

            yield cl


sqs = boto3.client('sqs',region_name='us-east-1', aws_access_key_id=os.environ["AMAZON_ACCESS_KEY_ID"],aws_secret_access_key=os.environ['AMAZON_SECRET_ACCESS_KEY'])

def delete_from_sqs(message):
    sqs.delete_message(QueueUrl=INSERT_QUEUE_URL_HERE,ReceiptHandle=message['ReceiptHandle'])
      messageBody = message['Body']
      messageBodyMd5 = hashlib.md5(messageBody.encode()).hexdigest()
      if messageBodyMd5 == message['MD5OfBody']:
      c = (json.loads(messageBody))
      try:
        already_seen_flag = c['data']['classification']['metadata']['subject_selection_state']['already_seen']
      except Exception as exep:
    #This error causes occasional print-out of 'subject_selection state', when the 'already_seen' flag isn't present in the AWS message.
        print(exep)
        already_seen_flag = True
        with open(config_swap().keyerror_list_path, 'r') as g:
            keyerror_list=eval(g.read())
        keyerror_list[2]+=1
        with open(config_swap().keyerror_list_path, 'w') as h:
            h.write(str(keyerror_list))
      try:
        user_flag = int(c['data']['classification']['user_id'])
      except Exception as excep_2:
        try:
            user_flag = c['data']['classification']['metadata']['session']
        except Exception as excep_4:
            print(excep_4)
            with open(config_swap().keyerror_list_path, 'r') as m:
                keyerror_list=eval(m.read())
            keyerror_list[3]+=1
            with open(config_swap().keyerror_list_path, 'w') as n:
                n.write(str(keyerror_list))
            print('User not logged on: cant use session flag so ignoring classification')
    #******FIX/CHECK THE NEXT LINE WORKS AS ITS SUPPOSED TO!!!!!!!!!*******
            return []
      cl = {
            'id': int(c['classification_id']),
            'subject': int(c['subject_id']),
            'project': int(c['data']['classification']['project_id']),
            'workflow': c['data']['classification']['workflow_id'],
            'annotations': c['data']['classification']['annotations'],
            # Assumes that extractor will handle user ID
            'user': user_flag,
            'already_seen': already_seen_flag,
            'classification_time':c['data']['classification']['created_at']}
      return cl

class SQSExtractor(Extractor):

    @classmethod
    def print_new_messages_in_table(self,response):
        if 'Messages' in response:
            r_db = DF()
            for message_0 in response['Messages']:
                # extract message body expect a JSON formatted string any information required to deduplicate the 
                # message should be present in the message body
                messageBody_0 = message_0['Body']
                # verify message body integrity
                messageBodyMd5_0 = hashlib.md5(messageBody_0.encode()).hexdigest()
                if messageBodyMd5_0 == message_0['MD5OfBody']:
                    rm_i = json.loads(messageBody_0)
                    rm_ii = {'Body_id':str(rm_i['id']),
                             'Classification_id':str(rm_i['classification_id']),
                             'Subj_id':str(rm_i['subject_id']),
                             'User_id':str(rm_i['data']['classification']['user_id']),
                             'Session_id':rm_i['data']['classification']['metadata']['session'][0:5]+'...'+\
                                          rm_i['data']['classification']['metadata']['session'][-5:],
                             'created_at':rm_i['data']['classification']['created_at'],
                             'updated_at':rm_i['data']['classification']['updated_at'],
                             'selected_at':rm_i['data']['classification']['metadata']['subject_selection_state']['selected_at']}
                    r_db=r_db.append(rm_ii,ignore_index=True)
                else:
                    print('Something not working')
                    print(messageBodyMd5_0)
            print('Raw classifications:')
            print(r_db)
        else:
            print(f'No messages: {response}')

    @classmethod
    def print_processed_messages_in_table(self,response):
        #[{'id': 515208555, 'subject': 92365948, 'project': 21834, 'workflow': 25011, 'annotations': {'T1': [{'task': 'T1', 'value': []}]}, 'user': 2336992, 'already_seen': False, 'classification_time': '2023-10-09T10:32:20.549Z'}]
        try:
            r_db = DF()
            for message_i in response:
                rm_ii = {'Classification_id':str(message_i['id']),
                         'Subj_id':str(message_i['subject']),
                         'User_id':str(message_i['user']),
                         'created_at':str(message_i['classification_time'])}
                r_db=r_db.append(rm_ii,ignore_index=True)
            print('Processed classifications, having removed duplicates:')
            print(r_db)
        except Exception as ex_proc_messages:
            print('Exception when printing processed messages:',ex_proc_messages)

    @classmethod
    def get_classifications(cls, queue_url):
        for c in SQSClient.extract(queue_url):
          try:
            already_seen_flag = c['data']['classification']['metadata']['subject_selection_state']['already_seen']
          except Exception as excep_3:
            print('Error here, exception:')
            print(exep_3)
            already_seen_flag = True
            with open(config_swap().keyerror_list_path, 'r') as g:
              keyerror_list=eval(g.read())
            keyerror_list[2]+=1
            with open(config_swap().keyerror_list_path, 'w') as h:
              h.write(str(keyerror_list))
          try:
            user_flag = int(c['data']['classification']['user_id'])
          except:
            try:
              user_flag = c['data']['classification']['metadata']['session']
            except:
              with open(config_swap().keyerror_list_path, 'r') as m:
                keyerror_list=eval(m.read())
              keyerror_list[3]+=1
              with open(config_swap().keyerror_list_path, 'w') as n:
                n.write(str(keyerror_list))
              print('User not logged on: cant use session flag so ignoring classification')
              continue
          cl = {
                'id': int(c['classification_id']),
                'subject': int(c['subject_id']),
                'project': int(c['data']['classification']['project_id']),
                'workflow': c['data']['classification']['workflow_id'],
                'annotations': c['data']['classification']['annotations'],
                # Assumes that extractor will handle user ID
                'user': user_flag,
                'already_seen': already_seen_flag,
                'classification_time':c['data']['classification']['created_at']}
          yield cl

    @classmethod
    def sqs_retrieve_fast(self):
        messages_to_be_deleted= []
        return_responses = []
        self.sqs = boto3.client('sqs',region_name='us-east-1',\
                                aws_access_key_id=os.environ["AMAZON_ACCESS_KEY_ID"],\
                                aws_secret_access_key=os.environ['AMAZON_SECRET_ACCESS_KEY'])
        queue_url=Config.instance().sqs_queue
        for k in range(config_swap().number_of_batches_of_10):
            print(f'Batch {k}')
            Queue_attributes_dict = self.sqs.get_queue_attributes(QueueUrl = queue_url,AttributeNames=['All'])['Attributes']
            print('Queue attributes:',{k:Queue_attributes_dict[k] for k in ['ApproximateNumberOfMessages',\
                                                                            'ApproximateNumberOfMessagesNotVisible',\
                                                                            'ApproximateNumberOfMessagesDelayed']})
            #Returns a list of all the classifications made, and corresponding metadata:
            response = self.sqs.receive_message(
                QueueUrl=queue_url,
                AttributeNames=['SentTimestamp', 'MessageDeduplicationId'],
                MaxNumberOfMessages=10,  # Allow up to 10 messages to be received
                MessageAttributeNames=['All'],
                # Allows the message to be retrieved again after 1hr
                VisibilityTimeout=3600,
                # Wait at most 20 seconds for an extract enables long polling
                WaitTimeSeconds=20)
#            print(response)
            receivedMessageIds = []
            ##########
            try: 
                self.print_new_messages_in_table(response)
            except Exception as ex_message:
                print(f'Cannot print messages: {ex_message}')
            ###########
            try:
                messages_to_be_deleted.extend(list(response['Messages']))
                return_responses.extend(list(response['Messages']))
            except:
                pass
        print('Total number of responses in batch:' + str(len(return_responses)))
        try:
            N_proc_to_use = np.max([np.min([16,len(return_responses)]),1])
            print(f'Using {N_proc_to_use} processes')
            ST_MP = time.time()
            with Pool(N_proc_to_use) as pool:
                return_responses = pool.map(delete_from_sqs, messages_to_be_deleted)
                #Below is code to remove classifications which have already been made (i.e. their classification_id, subject_id AND user_id all match a previous classification), preventing them from being passed to SWAP.
                try:
                    full_cl_id_list = np.array(retrieve_list(config_swap().classification_id_path))
                    new_cl_id_list = np.array(['_T_'.join([str(elem['id']),
                                                str(elem['subject']),
                                                str(elem['user'])]) for elem in return_responses])
                    new_cl_indx = np.where(~np.isin(new_cl_id_list,full_cl_id_list))[0]
                    repeated_cl_indx = np.where(np.isin(new_cl_id_list,full_cl_id_list))[0]
                    save_list(full_cl_id_list.tolist()+new_cl_id_list[new_cl_indx].tolist(),
                              config_swap().classification_id_path)
                    repeated_cl_time = [elem['classification_time'] for elem in \
                                        np.array(return_responses)[repeated_cl_indx]]
                    new_cl_time  = [elem['classification_time'] for elem in \
                                        np.array(return_responses)[new_cl_indx]]
                    print('REPEATED CL TIME',repeated_cl_time)
                    print('REPEATED SUBJ',[elem['subject'] for elem in np.array(return_responses)[repeated_cl_indx]])
                    print('NEW CL TIME',new_cl_time)
                    return_responses = np.array(return_responses)[new_cl_indx].tolist()
                    #print('Return responses',return_responses)
                    print('Total number of new responses in batch: ' + str(len(return_responses)))
                except Exception as ex1:
                    print(ex1)
                    with open(config_swap().keyerror_list_path, 'r') as m:
                        keyerror_list=eval(m.read())
                        keyerror_list[9]+=1
                    with open(config_swap().keyerror_list_path, 'w') as n:
                        n.write(str(keyerror_list))
                    print('KEY_ERROR_9 HERE')
                    pass
            print(f'Time to MP: {time.time()-ST_MP}')
            try: 
                self.print_processed_messages_in_table(return_responses)
            except Exception as ex_proc_message:
                print(f'Cannot print processed messages: {ex_proc_message}')
            return list(return_responses)
        except Exception as excep_5:
            print('Exception here:')
            print(excep_5)
            print(traceback.format_exc())
            with open(config_swap().keyerror_list_path, 'r') as g:
                 keyerror_list=eval(g.read())
            keyerror_list[5]+=1
            with open(config_swap().keyerror_list_path, 'w') as h:
                h.write(str(keyerror_list))
            return []

