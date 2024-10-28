

"""
Get classifications from Panoptes or SQS Queue
"""
from caesar_external.utils.caesar_utils import Client, SQSClient,UniqueMessage
from config import Config as config_swap
from caesar_external.data import Config
from pandas import DataFrame as DF
from multiprocessing import Pool
import pandas as pd
import numpy as np
import traceback
import hashlib
import logging
import boto3
import json
import time
import sys
import os
logger = logging.getLogger(__name__)
with open('../caesar_external/data/AWS_config.json') as f:
   AWS_config_data = json.load(f)

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
    list_db = pd.DataFrame({'Column 1': list_total}) #Dont change this column title unless change those in def retrieve_list() too.
    list_db.to_csv(list_path, index=False)

sqs = boto3.client('sqs',region_name='us-east-1',
                   aws_access_key_id=os.environ["AMAZON_ACCESS_KEY_ID"],
                   aws_secret_access_key=os.environ['AMAZON_SECRET_ACCESS_KEY'])

def process_and_delete_from_sqs(message):
      messageBody = message['Body']
      messageBodyMd5 = hashlib.md5(messageBody.encode()).hexdigest()
      if messageBodyMd5 == message['MD5OfBody']:
        sqs.delete_message(QueueUrl=AWS_config_data['sqs_queue'],
                           ReceiptHandle=message['ReceiptHandle'])
      c = json.loads(messageBody)
      try:
        user_flag = int(c['data']['classification']['user_id'])
      except Exception as excep_2:
        try:
            user_flag = c['data']['classification']['metadata']['session']
        except Exception as excep_4:
            '''Key Error 3 indicates if there is a problem with retrieving the user id from the SQS message.'''
            print('KEY_ERROR_3_HERE',excep_4)
            with open(config_swap().keyerror_list_path, 'r') as m:
                keyerror_list=eval(m.read())
            keyerror_list[3]+=1
            with open(config_swap().keyerror_list_path, 'w') as n:
                n.write(str(keyerror_list))
            print("User not logged on: can't use session flag so ignoring classification")
            return []
      cl = {
            'id': int(c['classification_id']),
            'subject': int(c['subject_id']),
            'project': int(c['data']['classification']['project_id']),
            'workflow': c['data']['classification']['workflow_id'],
            'annotations': c['data']['classification']['annotations'],
            'user': user_flag, # Assumes that extractor will handle user ID
            'classification_time':c['data']['classification']['created_at']}
      return cl

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
                    # r_db=r_db.append(rm_ii,ignore_index=True)
                    r_db = pd.concat([r_db,DF(rm_ii,index=[0])]).reset_index(drop=True)
                else:
                    print('Something not working')
                    print(messageBodyMd5_0)
            print('Raw classifications:')
            print(r_db)
        else:
            print(f'No messages: {response}')

    @classmethod
    def print_processed_messages_in_table(self,response):
        try:
            r_db = DF()
            for message_i in response:
                rm_ii = {'Classification_id':str(message_i['id']),
                         'Subj_id':str(message_i['subject']),
                         'User_id':str(message_i['user']),
                         'created_at':str(message_i['classification_time'])}
                # r_db=r_db.append(rm_ii,ignore_index=True)
                r_db = pd.concat([r_db,DF(rm_ii,index=[0])]).reset_index(drop=True)
            print('Processed classifications, having removed duplicates:')
            print(r_db)
        except Exception as ex_proc_messages:
            print('Exception when printing processed messages:',ex_proc_messages)

    @classmethod
    def get_classifications(cls, queue_url):
        for c in SQSClient.extract(queue_url):
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
                QueueUrl = queue_url,
                AttributeNames=['SentTimestamp', 'MessageDeduplicationId'],
                MaxNumberOfMessages=10,  # Allow up to 10 messages to be received
                MessageAttributeNames=['All'],
                VisibilityTimeout=3600,# Allows the message to be retrieved again after 1hr
                WaitTimeSeconds=20 # Wait at most 20 seconds for an extract enables long polling
                )
            receivedMessageIds = []
            try: 
                self.print_new_messages_in_table(response)
            except Exception as ex_message:
                print(f'Cannot print messages: {ex_message}')
            try:
                messages_to_be_deleted.extend(list(response['Messages']))
                return_responses.extend(list(response['Messages']))
            except:
                pass
        print('Total number of responses in batch:' + str(len(return_responses)))
        try:
            #Assuming the code is being run with multiple processors e.g. on a computing cluster, this uses between 1-16 processors 
            #(if running locally, N_proc_to_use can just be set to e.g. 1).
            N_proc_to_use = np.max([np.min([16,len(return_responses)]),1])
            print(f'Using {N_proc_to_use} processes')
            with Pool(N_proc_to_use) as pool:
                return_responses = pool.map(process_and_delete_from_sqs, messages_to_be_deleted)
            try: 
                self.print_processed_messages_in_table(return_responses)
            except Exception as ex_proc_message:
                print(f'Cannot print processed messages: {ex_proc_message}')
            return list(return_responses)
        except Exception as excep_5:
            print('Exception here:', excep_5)
            print(traceback.format_exc())
            """Key error 5 indicates there is a problem when trying to delete messages from SQS queue in Caesar Extractor"""
            with open(config_swap().keyerror_list_path, 'r') as g:
                 keyerror_list=eval(g.read())
            keyerror_list[5]+=1
            with open(config_swap().keyerror_list_path, 'w') as h:
                h.write(str(keyerror_list))
            return []

