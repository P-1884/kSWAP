
import panoptes_client as pan
from panoptes_client.panoptes import PanoptesAPIException
from caesar_external.data import Config
import os
import hashlib
import json
import boto3
import time
import logging
logger = logging.getLogger(__name__)

"""
Utils to interact with Zooniverse...
"""


class Client:

    _instance = None

    def __init__(self):
        logger.info('Authenticating with Caesar...')
        config = Config.instance()
        kwargs = {
            'endpoint': config.login_endpoint(),
        }

        if config.auth_mode == 'api_key':
            kwargs.update({
                'client_id': config.client_id,
                'client_secret': config.client_secret,
            })
        elif config.auth_mode == 'interactive':
            kwargs.update({
                'login': 'interactive',
            })
        elif config.auth_mode == 'environment':
            # panoptes client will handle getting environment variables
            # for authentication.
            # keeping this here for clarity
            pass

        self.pan = pan.Panoptes(**kwargs)

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def extract(cls, project, last_id):
        cls.instance()
        kwargs = {
            'scope': 'project',
            'project_id': project,
        }
        if last_id:
            kwargs.update({'last_id': last_id})
        return pan.Classification.where(**kwargs)

    @classmethod
    def reduce(cls, subject, data):
        """
        PUT subject score to Caesar
        """
        config = Config._config
        pan = cls.instance().pan

        endpoint = config.caesar_endpoint()
        path = config.workflow_path()

        body = {
            'reduction': {
                'subject_id': subject,
                'data': data
            }
        }

        try:
            logger.debug('endpoint => {}, path => {}'.format(endpoint, path))
            r = pan.put(endpoint=endpoint, path=path, json=body)
            return r
        except PanoptesAPIException as e:
            print('Failed to send reduction for {}: {}'.format(subject, e))
        except json.decoder.JSONDecodeError as e:
            print('Error decoding JSON - Likely issue with endpoint')

        return None

class UniqueMessage(object):

    def __init__(self, message):
        logger.debug(message)
        self.classification_id = int(message['classification_id'])
        self.message = message

    def __eq__(self, other):
        return self.classification_id == other(self.classification_id)

    def __hash__(self):
        return hash(self.classification_id)

class SQSClient(Client):

    def __init__(self):
        super().__init__()
        # Create SQS client
        self.sqs = boto3.client('sqs',region_name='us-east-1',
    aws_access_key_id=os.environ["AMAZON_ACCESS_KEY_ID"],aws_secret_access_key=os.environ['AMAZON_SECRET_ACCESS_KEY'])
    @classmethod
    def extract(cls, queue_url):
        return cls.instance().sqs_retrieve(queue_url)[0]
    @classmethod
    def extract_fast(cls,queue_url):
        return cls.instance().sqs_retrieve_fast(queue_url)

    def sqs_retrieve(self, queue_url):
        response = self.sqs.receive_message(
            QueueUrl=queue_url,
            AttributeNames=[
                'SentTimestamp', 'MessageDeduplicationId'
            ],
            MaxNumberOfMessages=10,  # Allow up to 10 messages to be received
            MessageAttributeNames=[
                'All'
            ],
            # Allows the message to be retrieved again after 40s
            VisibilityTimeout=20,
            # Wait at most 20 seconds for an extract enables long polling
            WaitTimeSeconds=20
        )

        receivedMessageIds = []
        receivedMessages = []
        uniqueMessages = set()
        # Loop over messages
        if 'Messages' in response:
            for message in response['Messages']:
                # extract message body expect a JSON formatted string
                # any information required to deduplicate the message should be
                # present in the message body
                messageBody = message['Body']
                # verify message body integrity
                messageBodyMd5 = hashlib.md5(messageBody.encode()).hexdigest()

                if messageBodyMd5 == message['MD5OfBody']:
                    receivedMessages.append(json.loads(messageBody))
                    receivedMessageIds.append(
                        receivedMessages[-1]['classification_id'])
                    uniqueMessages.add(UniqueMessage(receivedMessages[-1]))
                    # the message has been retrieved successfully - delete it.
                    self.sqs_delete(queue_url, message['ReceiptHandle'])


        logger.debug('Num Duplicated IDS: {}'.format(
            len(receivedMessages) - len(uniqueMessages)))

        messages = [m.message for m in uniqueMessages]
        return messages, receivedMessages, receivedMessageIds

    def sqs_delete(self, queue_url, receipt_handle):
        self.sqs.delete_message(
            QueueUrl=queue_url,
            ReceiptHandle=receipt_handle
        )


