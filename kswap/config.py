class Config(object):
  def __init__(self):
    # panoptes
    self.project = 6968
    self.workflow = 8878
    # data paths
    self.swap_path = './'
    self.data_path = self.swap_path+'/data/'
    self.db_name = 'swap.db'
    self.db_path = './data/'

    self.user_default = {'0':0.5, '1':0.5}
    self.label_map = {'0':0, '1':1}
    self.classes = ['0', '1']
    self.p0 = 5.e-4
    self.gamma = 1
    self.thresholds = (1.e-5, 1)
    self.lower_retirement_limit=4
    self.retirement_limit = 30

#File Paths:
    self.tuples_path = '/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/tuples_data'
    self.retired_items_path='/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/retired_list'
    self.golds_path= './data/des_golds_beta_test_25jan.csv'
#Path below might be redundant?
    self.kswap_path='/Users/hollowayp/Documents/GitHub/kSWAP/kswap'
#When plotting trajectories, there is an additional path (on line 2 of kswap_plots.py) which needs to be updated to the location of the config file (found at '...kSWAP/kswap/config.py')
