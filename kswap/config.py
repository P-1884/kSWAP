class Config(object):
  def __init__(self):
    # panoptes
    self.project = 6968
#TEMPORARILY CHANGING THE WORKFLOW TO HSC TO RUN THE HSC_CLASSIFICATIONS: NEED TO CHANGE THIS BACL FOR SW DES:
#    self.workflow = 8878
    self.workflow = 6605 #HSC workflow 5374 was the beta-test - these are ignored by swap.
    # data paths
    self.swap_path = './'
    self.data_path = self.swap_path+'/data/'
    self.shuffle_time = 0
#    self.db_name = 'swap_shuffled_' + str(self.shuffle_time) + '.db'
    self.db_name = 'swap_hsc_full_exclude_not_logged_on.db'
    self.db_path = './data/'

    self.user_default = {'0':0.5, '1':0.5}
    self.label_map = {'0':0, '1':1}
    self.classes = ['0', '1']
    self.p0 = 5.e-4
    self.gamma = 1
    self.thresholds = (1.e-5, 1)
    self.lower_retirement_limit=4
#SHOULD BE 30:
    self.retirement_limit = 30

#File Paths:
    self.tuples_path = '/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/tuples_data'
    self.retired_items_path='/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/retired_list'
    self.golds_path= './data/HSC_Classifications_and_Golds/golds.csv'
#    self.classification_path = './data/HSC_Classifications_and_Golds/swhsc-6605-classification_060518_0317.csv'
    self.classification_path = './data/HSC_Classifications_and_Golds/space-warps-hsc-classifications.csv'
    self.hard_sims_path = None
#    self.hard_sims_path = './data/des_golds_beta_test_25jan.csv'
#Path below might be redundant?
    self.kswap_path='/Users/hollowayp/Documents/GitHub/kSWAP/kswap'
#When plotting trajectories, there is an additional path (on line 2 of kswap_plots.py) which needs to be updated to the location of the config file (found at '...kSWAP/kswap/config.py')
