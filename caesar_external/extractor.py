

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

from multiprocessing import Pool

sqs = boto3.client('sqs',region_name='us-east-1', aws_access_key_id=os.environ["AMAZON_ACCESS_KEY_ID"],aws_secret_access_key=os.environ['AMAZON_SECRET_ACCESS_KEY'])

def delete_from_sqs(message):
  messageBody = message['Body']
  messageBodyMd5 = hashlib.md5(messageBody.encode()).hexdigest()
  if messageBodyMd5 == message['MD5OfBody']:
    sqs.delete_message(QueueUrl=INSERT_QUEUE_URL_HERE,ReceiptHandle=message['ReceiptHandle'])
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
        'id': int(c['id']),
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
                'id': int(c['id']),
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
        for k in range(config_swap().number_of_batches_of_10):
            self.sqs = boto3.client('sqs',region_name='us-east-1',\
                                    aws_access_key_id=os.environ["AMAZON_ACCESS_KEY_ID"],\
                                    aws_secret_access_key=os.environ['AMAZON_SECRET_ACCESS_KEY'])
            queue_url=Config.instance().sqs_queue
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
            receivedMessages = []
            uniqueMessages = set()
            ##########
            uniqueMessages_0 = set()

            # Loop over messages
            print('Loopy')
#            print(response)
            if 'Messages' in response:
                for message_0 in response['Messages']:
                    print(1)
                    # extract message body expect a JSON formatted string
                    # any information required to deduplicate the message should be
                    # present in the message body
                    messageBody_0 = message_0['Body']
                    # verify message body integrity
                    messageBodyMd5_0 = hashlib.md5(messageBody_0.encode()).hexdigest()
                    if messageBodyMd5_0 == message_0['MD5OfBody']:
                        receivedMessages.append(json.loads(messageBody_0))
                        uniqueMessages_0.add(UniqueMessage(receivedMessages[-1]))
            logger.debug('Num Duplicated IDS: {}'.format(
            len(receivedMessages) - len(uniqueMessages)))
            ###########
            #print(response)
            try:
                messages_to_be_deleted.extend(list(response['Messages']))
                return_responses.extend(list(response['Messages']))
            except:
                pass
            try:
                for rep_i in response['Messages']:
                    print('Session ID',eval(rep_i['Body'].replace('false','False').replace('true','True').replace('null','"Null"'))['data']['classification']['metadata']['session'])
                    print('User id: ',eval(rep_i['Body'].replace('false','False').replace('true','True').replace('null','"null"'))['user_id'])
                    print('Subject id: ',eval(rep_i['Body'].replace('false','False').replace('true','True').replace('null','"null"'))['subject_id'])

            except Exception as ex1:
                pass
        print('Total number of responses in batch:' + str(len(return_responses)))
        try:
            with Pool(np.max([np.min([4,len(return_responses)]),1])) as pool:
                return_responses = pool.map(delete_from_sqs, messages_to_be_deleted)
#          with open('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/Classification_list.txt', 'r') as q:
#            Classification_list=eval(q.read())
#          for i in range(len(return_responses)):
#            if return_responses[i]['id'] not in Classification_list:
#              Classification_list.append(return_responses[i]['id'])
#            else:
#              with open('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/KeyError_list.txt', 'r') as m:
#                keyerror_list=eval(m.read())
#              keyerror_list[6]+=1
#              with open('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/KeyError_list.txt', 'w') as n:
#                n.write(str(keyerror_list))
#              print('AWS DELETION ERROR HERE')
#              print(return_responses[i]['id'])
#          with open('/Users/hollowayp/Documents/GitHub/kSWAP/kswap/Classification_list.txt', 'w') as r:
#            r.write(str(Classification_list))
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


