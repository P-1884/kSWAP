import sys
sys.path.append('../kswap/')
sys.path.append('../')
from swap import SWAP

'''
NOTE:  There is nothing yet preventing user scores being updated multiple times if a given user 
saw the same gold subjects more than once in test_offline or test_online. This is resolved for AWS swap.
'''
def test_offline():
  #Processes classifications from a csv data export. User skills are updated first, then the subject scores are 
  #updated according to the (fixed) user skills.
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
  #Processes classifications from a csv data export. User skills are updated simultaneously with the subject scores.
  from config import Config
  swap = SWAP(config=Config())
  swap = swap.load()
  swap.run_online(Config().golds_path,Config().classification_path,Config().hard_sims_path)
  swap.save()
  del swap
  swap = SWAP(config=Config())
  swap = swap.load()

def test_caesar():
  #Processes classifications retrieved in real-time from an AWS queue.
  from config import Config
  swap = SWAP(config=Config())
  swap = swap.load()
  swap.run_caesar(Config().golds_path,Config().hard_sims_path)
  swap.save()
  del swap
  swap = SWAP(config=Config())
  swap = swap.load()

def main():
  print('Hello World')
#To run swap, uncomment the required version of swap here:
#  test_offline() #Offline swap from csv
#  test_online() #Online swap from csv
#  test_caesar() #AWS Swap

if __name__ == '__main__':
  main()
