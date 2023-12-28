# kSWAP
Small adjustments from the original (see https://github.com/zooniverse/kSWAP), now adapting for SpaceWarps binary classifications. 
# Caesar External
Included here for completeness, adapted from https://github.com/miclaraia/caesar_external

## Setup
A few things need to be configured before this code package will run. 
### The config file:
This is located at kswap/config.py. The config folder requires details of the Zooniverse project/workflow ID, along with
selected paths to files SWAP requires/will generate. 
SWAP generates X files when running:
1) A database of classifications, containing details of user skills and subject scores. In AWS SWAP this is updated in 
real-time.
2) List of all subjects which have been retired (located at 'retired_items_path').
3) List of IDs and retirement times of subjects which have been retired due to reaching the upper classification-number limit (located at Nclass_retirement_path).
4) List of IDs and retirement times of subjects which have been retired due to reaching the upper or lower score threshold (located at Nthres_retirement_path).
5) Bug Report (located at keyerror_list_path): This is a list of bug occurrences to help with debugging (primarily for AWS SWAP). The numbers are updated (>0) each time abnormal behaviour is detected. 
### AWS Queue

### And Finally
Once the above has been completed, run SWAP by uncommenting one of the functions in 'main' within examples/run.py.

