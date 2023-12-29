import os

class Config(object):
  def __init__(self):
    self.project = XXX #Insert Zooniverse project ID here.
    self.workflow = XXX #Insert Zooniverse Workflow ID here.
    self.swap_path = './'
    self.data_path = self.swap_path+'/data/'
    self.db_name = 'swap_database.db' #Filename to save the swap database as.
    self.db_path = './data/'

    self.user_default = {'0':0.5, '1':0.5} #Initial user skill
    self.label_map = {'0':0, '1':1}
    self.classes = ['0', '1']
    self.p0 = 5e-4 #Initial subject score
    self.gamma = 1
    self.thresholds = (1.e-5, 1) #Lower and upper retirement subject score thresholds
    self.lower_retirement_limit=5 #Number of classifications before a subject can be retired (>=N)
    self.retirement_limit = 30 #Maximum number of classifications a test subject can receive before retirement

#File Paths: 
    #File-path for list of subjects which have been retired:
    self.retired_items_path= '../kswap/retired_items_list.txt' #This should start with 'Column 1' as its first line.
    #File-path for list of subjects which have been retired because they reached the classification number limit:
    self.Nclass_retirement_path = '../kswap/Nclass_retirement_list.txt'  #This should start with 'Column 1' as its first line.
    #File-path for list of subjects which have been retired because they reached the lower or upper score thresholds:
    self.Nthres_retirement_path = '../kswap/Nthres_retirement_list.txt' #This should start with 'Column 1' as its first line.
    #File-path for where the gold-subject csv is stored:
    self.golds_path = './data/golds_file.csv' #This is currently a placeholder file.
    #If importing classifications from a csv file, this should be the path to that file, otherwise None:
    self.classification_path = None
    #File-path to the csv file for identifying the hard sims (if any), otherwise None:
    self.hard_sims_path = None
    self.kswap_path = '../kswap/'
    self.examples_path = '../examples'

#When plotting trajectories, there is an additional path (on line 2 of kswap_plots.py) which needs to be updated to the location of the config file (found at '...kSWAP/kswap/config.py')
    
    #File-path for error-flags. The file should originally be a single line of [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #These numbers are updated (+1) as swap runs, each time abnormal behaviour is detected. 
    self.keyerror_list_path = '../kswap/KeyError_list.txt'
    #File-path for number of classifications retrieved. The file should originally be a single line of [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    #These numbers are updated (+1) as swap runs, indicating the number of classifications retrieved from the AWS queue (0 to 10 in the list). 
    self.aws_list_path = '../kswap/AWS_list.txt'
    #This is the number of batches of 10 classifications retrieved from the queue at a time, before they're sent for processing by SWAP.
    self.number_of_batches_of_10 = 20 
#IMPORTANT NOTE: There is an additional config file within caesar_extractor (opt/anaconda3/lib/python3.8/site-packages/caesar_external/data/test_config2.json) which needs to be configured to the correct workflow etc.
