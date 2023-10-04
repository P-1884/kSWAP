import os

class Config(object):
  def __init__(self):
    # panoptes
    self.project = 21834#6968
#DES workflow:
    self.workflow = 25011#8878
#THIS (BELOW) IS FOR CHANGING THE WORKFLOW TO HSC TO RUN THE HSC_CLASSIFICATIONS:
#    self.workflow = 6605 #HSC workflow 5374 was the beta-test - these are ignored by swap.
    # data paths
    self.swap_path = './'
    self.data_path = self.swap_path+'/data/'
    self.shuffle_time = 0
#    self.db_name = 'swap_shuffled_' + str(self.shuffle_time) + '.db'
    self.db_name = 'swap_beta_test_db_Sept2023_caesar_testing.db'#_from_csv.db'#'swap_hsc_full_exclude_not_logged_on.db'
    self.db_path = './data/'

    self.user_default = {'0':0.5, '1':0.5}
    self.label_map = {'0':0, '1':1}
    self.classes = ['0', '1']
    self.p0 = 5e-4 #2.e-4 #Using HSC prior (April 2023) to run the HSC classifications. Have changed this from 5.e-4
    self.gamma = 1
    self.thresholds = (1.e-5, 1)
#SHOULD BE 4:
    self.lower_retirement_limit=5#14
#SHOULD BE 30:
    self.retirement_limit = 30

#File Paths: 
    self.retired_items_path=None#'./data/retired_list_Sept2023_test.txt'#This should start with 'Column 1' as its first line.
    self.Nclass_retirement_path =None# './data/retired_list_Sept2023_Nclass_test.txt'#This should start with 'Column 1' as its first line.
    self.Nthres_retirement_path = None#'./data/retired_list_Sept2023_Nthres_test.txt'#This should start with 'Column 1' as its first line.
    self.golds_path = './data/des_beta_test_golds.csv'
#    self.golds_path= './data/HSC_Classifications_and_Golds/golds.csv'
#    self.classification_path = './data/HSC_Classifications_and_Golds/swhsc-6605-classification_060518_0317.csv'
#    self.classification_path = './data/HSC_Classifications_and_Golds/space-warps-hsc-classifications.csv'
    self.classification_path = './data/space-warps-des-vision-transformer-classifications_beta_FINAL.csv'
    self.hard_sims_path = None
#    self.hard_sims_path = './data/des_golds_beta_test_25jan.csv'
    self.kswap_path = '/mnt/zfsusers/hollowayp/kSWAP/kswap/'
#    self.kswap_path='/Users/hollowayp/Documents/GitHub/kSWAP/kswap'
    self.examples_path = '/mnt/zfsusers/hollowayp/kSWAP/examples'
#    self.examples_path = '/Users/hollowayp/Documents/GitHub/kSWAP/examples'

#When plotting trajectories, there is an additional path (on line 2 of kswap_plots.py) which needs to be updated to the location of the config file (found at '...kSWAP/kswap/config.py')
    
#Extra files:
    self.keyerror_list_path = '/mnt/zfsusers/hollowayp/kSWAP/kswap/KeyError_list_Sept2023_test.txt'
    self.aws_list_path = '/mnt/zfsusers/hollowayp/kSWAP/kswap/AWS_list_Sept2023_test.txt'
#    self.keyerror_list_path = '/Users/hollowayp/Documents/GitHub/kSWAP/kswap/KeyError_list_Sept2023_test.txt'
#    self.aws_list_path = '/Users/hollowayp/Documents/GitHub/kSWAP/kswap/AWS_list_Sept2023_test.txt'
    self.number_of_batches_of_10 = 2 #Should be ~ 20, this is the number of batches of 10 classifications retrieved from the queue at a time, before they're sent for processing.

#IMPORTANT NOTE: There is an additional config file within caesar_extractor (opt/anaconda3/lib/python3.8/site-packages/caesar_external/data/test_config2.json) which needs to be configured to the correct workflow etc.
