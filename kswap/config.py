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
    self.thresholds = (1.e-7, 0.95)
    self.retirement_limit = 2
