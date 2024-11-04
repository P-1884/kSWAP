
# Overview
This package derives from two existing packages and implements AWS SWAP: real-time processing of binary classifications made on the Zooniverse platform. The original packages are:
### kSWAP
Adapted from https://github.com/zooniverse/kSWAP. 
### Caesar External
Included here for completeness, adapted from https://github.com/miclaraia/caesar_external. The key functions used here are found within caesar_external/extractor.py.

# Setup
A few things need to be configured before this code package will run. 

## Requirements
The following packages are required (available via pip):
- panoptes_client,
- boto3\
Once these packages have been installed correctly you should be able to run python3 run.py (after navigating to the examples folder) without errors. You should then configure the config file, as described below.

### The Config file:
This is located at kswap/config.py. The config folder requires details of the Zooniverse project/workflow ID, along with
selected paths to files SWAP requires/will generate. 
SWAP generates the following files when running:
1) A database of classifications, containing details of user skills and subject scores. In AWS SWAP this is updated in 
real-time.
2) List of all subjects which have been retired (located at 'retired_items_path').
3) List of IDs and retirement times of subjects which have been retired due to reaching the upper classification-number limit (located at 'Nclass_retirement_path').
4) List of IDs and retirement times of subjects which have been retired due to reaching the upper or lower score threshold (located at 'Nthres_retirement_path').
5) Bug Report (located at 'keyerror_list_path'): This is a list of bug occurrences to help with debugging (primarily for AWS SWAP). The numbers are updated (>0) each time abnormal behaviour is detected. The elements of the list correspond to the following bugs:\
[0]: Error in caesar_receive,\
[1]: Error in caesar_receive,\
[2]: Currently Redundant,\
[3]: Error when looking for user-id in Caesar Extractor,\
[4]: Error in sending subjects to panoptes for retirement,\
[5]: Error when trying to delete messages from SQS queue in Caesar Extractor,\
[6]: Error when saving retired list - retired_list path name may not be set?,\
[7]: Catch-all exception,\
[8]: User sees the same test subject again,\
[9]: Currently Redundant

### The Caesar Config file:
Located at kSWAP/caesar_external/data/AWS_config.json. This requires the Zooniverse project and workflow ID as well as the AWS url.

### AWS Queue
The AWS queue, which retrieves real-time classification data, needs to be configured to run AWS SWAP. Instructions for doing this are [here](https://docs.google.com/document/d/1kFpuq2QxfeXJRy6cIiAQgYiYt2Z2lr_OB5hIKPG246Y/edit?usp=sharing) (Credit: Zooniverse). This will generate two keys, AMAZON_ACCESS_KEY_ID and AMAZON_SECRET_ACCESS_KEY which are used by this package.

### And Finally
Once the above has been completed, run SWAP by uncommenting one of the functions in 'main' within examples/run.py. There is a 'quick_check' 
file which gives a quick summary of classifications so far (useful for when running AWS SWAP to check progress) and a 'SWAP_plots' file to plot trajectory plots of the classifications.\
Please report any bugs as Github Issues.
