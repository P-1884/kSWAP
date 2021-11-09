import sys
sys.path.insert(0, '../kswap')
from swap import SWAP
from kswap import kSWAP

#Loads the data from the databases, adds the golds to the subject class and applies these, processes classifications from the csv and updates the databases.
#I think this is redundant/duplication if using test_offline/online functions below?
def test_initialise():
  from config import Config
  swap = SWAP(config=Config())
  swap = swap.load()
  swap.get_golds('./data/golds.csv')
  swap.apply_golds('./data/swhsc-6605-classification_060518_0317.csv')
  swap.process_classifications_from_csv_dump('./data/swhsc-6605-classification_060518_0317.csv')
  swap.save()
  del swap
  swap = SWAP(config=Config())
  swap = swap.load()

#Loads data from databases, get/apply golds then processes classification from csv then updates the databases.
def test_offline():
  from offline_config import Config
  swap = SWAP(config=Config())
  swap = swap.load() #ie any previous user/subject details (otherwise gives default values to user/subject).
  swap.run_offline('./data/golds.csv',
                   './data/swhsc-6605-classification_060518_0317.csv')
  swap.save()
  del swap
  swap = SWAP(config=Config())
  swap = swap.load()

#Loads data from databases, gets golds, then applies golds/updates scores simultaneously.
def test_online():
  from online_config import Config
  swap = SWAP(config=Config())
  swap = swap.load()
  swap.run_online('./data/golds.csv',
                  './data/swhsc-6605-classification_060518_0317.csv')
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

  print(len(offline_user.history), len(online_user.history))
  print(offline_user.confusion_matrix, online_user.confusion_matrix)
  print(offline_user.user_score, online_user.user_score)
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
#  print('1: Initialise')
#  test_initialise()
#  print('2: Offline')
#  test_offline()
  print('3: Online')
  test_online()
#  print('4: Compare')
#  compare_offline_and_online_user_scores(user_id=1517738)

if __name__ == '__main__':
  main()
