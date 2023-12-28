import os

class Config(object):
  def __init__(self):
    # panoptes
    self.project = 21834#6968
    #DES-VT workflow:
    self.workflow = 25011#8878
    self.swap_path = './'
    self.data_path = self.swap_path+'/data/'
    self.shuffle_time = 0
    self.db_name = 'swap_DES_VT_Oct_2023.db'
    self.db_path = './data/'

    self.user_default = {'0':0.5, '1':0.5}
    self.label_map = {'0':0, '1':1}
    self.classes = ['0', '1']
    self.p0 = 5e-4
    self.gamma = 1
    self.thresholds = (1.e-5, 1)
#SHOULD BE 5:
    self.lower_retirement_limit=5#5
#SHOULD BE 30:
    self.retirement_limit = 50000

#File Paths: 
    self.retired_items_path='./data/retired_list_DES_VT_Oct_2023.txt' #This should start with 'Column 1' as its first line.
    self.Nclass_retirement_path ='./data/retired_list_DES_VT_Oct_2023_Nclass.txt' #This should start with 'Column 1' as its first line.
    self.Nthres_retirement_path = './data/retired_list_DES_VT_Oct_2023_Nthres.txt'#This should start with 'Column 1' as its first line.
    self.classification_id_path = '/mnt/zfsusers/hollowayp/kSWAP/kswap/Classification_id_DES_VT_Oct_2023.txt'#This should start with 'Column 1' as its first line.
    self.golds_path = './data/DES_VT_Oct_2023_golds.csv'
    self.classification_path = None #'./data/space-warps-des-vision-transformer-classifications_beta_FINAL.csv'
    self.hard_sims_path = None
    self.kswap_path = '/mnt/zfsusers/hollowayp/kSWAP/kswap/'
    self.examples_path = '/mnt/zfsusers/hollowayp/kSWAP/examples'

#When plotting trajectories, there is an additional path (on line 2 of kswap_plots.py) which needs to be updated to the location of the config file (found at '...kSWAP/kswap/config.py')
    
#Extra files:
    self.keyerror_list_path = '/mnt/zfsusers/hollowayp/kSWAP/kswap/KeyError_list_DES_VT_Oct_2023.txt'
    self.aws_list_path = '/mnt/zfsusers/hollowayp/kSWAP/kswap/AWS_list_DES_VT_Oct_2023.txt'
    self.number_of_batches_of_10 = 20 #Should be ~ 20, this is the number of batches of 10 classifications retrieved from the queue at a time, before they're sent for processing.

#IMPORTANT NOTE: There is an additional config file within caesar_extractor (opt/anaconda3/lib/python3.8/site-packages/caesar_external/data/test_config2.json) which needs to be configured to the correct workflow etc.
