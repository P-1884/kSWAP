import sys
sys.path.insert(0, '../kswap')
from swap import SWAP
from kswap import kSWAP

#Loads data from databases, get/apply golds then processes classification from csv then updates the databases.
def test_offline():
  from config import Config
  swap = SWAP(config=Config())
  swap = swap.load() #ie any previous user/subject details (otherwise gives default values to user/subject).
  swap.run_offline(Config().golds_path,
                     Config().classification_path,Config().hard_sims_path)
  swap.save()
  del swap
  swap = SWAP(config=Config())
  swap = swap.load()

#Loads data from databases, gets golds, then applies golds/updates scores simultaneously.
def test_online():
  from config import Config
  swap = SWAP(config=Config())
  swap = swap.load()
  swap.run_online(Config().golds_path,Config().classification_path,Config().hard_sims_path)
  #/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/HSC_Classifications_and_Golds/swhsc-6605-classification_060518_0317.csv
  #/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/alpha_test_classifications_cropped.csv
  swap.save()
  del swap
  swap = SWAP(config=Config())
  swap = swap.load()

def test_shuffled_online():
  from config import Config
  swap = SWAP(config=Config())
  swap = swap.load()
  swap.run_shuffled_online(Config().golds_path,Config().classification_path,Config().shuffle_time,Config().hard_sims_path)
  #/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/HSC_Classifications_and_Golds/swhsc-6605-classification_060518_0317.csv
  #/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/alpha_test_classifications_cropped.csv
  swap.save()
  del swap
  swap = SWAP(config=Config())
  swap = swap.load()

def test_caesar():
  from config import Config
  swap = SWAP(config=Config())
  swap = swap.load()
  swap.run_caesar(Config().golds_path,Config().hard_sims_path)
  swap.save()
  del swap
  swap = SWAP(config=Config())
  swap = swap.load()

def compare_offline_and_online_user_scores(user_id):
  import matplotlib.pyplot as plt
  
  from offline_config import Config as OfflineConfig
  offline_swap = SWAP(config=OfflineConfig())
  offline_swap = offline_swap.load()
  offline_user = offline_swap.users[user_id]
  
  from online_config import Config as OnlineConfig
  online_swap = SWAP(config=OnlineConfig())
  online_swap = online_swap.load()
  online_user = online_swap.users[user_id]

  #print(len(offline_user.history), len(online_user.history))
  #print(offline_user.confusion_matrix, online_user.confusion_matrix)
  #print(offline_user.user_score, online_user.user_score)
  assert len(offline_user.history) == len(online_user.history)
  plt.plot(range(len(offline_user.history)),
           [h[1]['1'] for h in offline_user.history],
           color='#26547C',
           label='\'Yes\' offline')
  plt.plot(range(len(offline_user.history)),
           [h[1]['0'] for h in offline_user.history],
           color='#EF476F',
           label='\'No\' offline')
  plt.plot(range(len(online_user.history)),
           [h[1]['1'] for h in online_user.history],
           color='#26547C',
           ls='--',
           label='\'Yes\' online')
  plt.plot(range(len(online_user.history)),
           [h[1]['0'] for h in online_user.history],
           color='#EF476F',
           ls='--',
           label='\'No\' online')
  plt.xlim(0,len(offline_user.history))
  plt.ylim(0,1)
  plt.xlabel('number of classifications')
  plt.ylabel('user score')
  plt.legend(loc='best')
  plt.show()

def main():

  ### SWAP tests
#  print('2: Offline')
#  test_offline()
  print('Starting Iteration')
#  test_online()
#  test_online()
  test_shuffled_online()
#  test_caesar()
  print('Finished Iteration')
#  test_online()
#  print('4: Compare')
#  compare_offline_and_online_user_scores(user_id=1517738)

if __name__ == '__main__':
  main()
