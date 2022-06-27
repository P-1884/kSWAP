import matplotlib.patches as mpatches
import matplotlib.pyplot as pl
import multiprocess as mp
import matplotlib as mpl
import pandas as pd
import numpy as np
import datetime
import sqlite3
import time
import csv
import sys
from pandas import read_sql_query, read_sql_table
from scipy.optimize import curve_fit as cf
from collections import Counter
from scipy import interpolate
from matplotlib import cm,colors
from tqdm import tqdm

sys.path.append('/Users/hollowayp/Documents/GitHub/kSWAP/kswap')

def thresholds_setting():
    from config import Config as config_0
    sys.path.insert(0, '../kswap')
    p_real = config_0().thresholds[1]
    p_bogus = config_0().thresholds[0]
    return [p_real,p_bogus]
    
def prior_setting():
    from config import Config as config_0
    return config_0().p0

def printer(q,N):
  for p in range(10):
      if q == int(p*N/10):
        print(str(10*p)+'%')

def read_sqlite(dbfile):
    with sqlite3.connect(dbfile) as dbcon:
      tables = list(read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
      out = {tbl : read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}
    return out

def date_time_convert(row_created_at):
  s_i=[]
  for s in range(len(row_created_at)):
    if row_created_at[s]=='-' or row_created_at[s]==':':
      s_i.append(s)
  dt_i = datetime.datetime(int(row_created_at[0:s_i[0]]), int(row_created_at[s_i[0]+1:s_i[1]]), int(row_created_at[s_i[1]+1:s_i[1]+3]),int(row_created_at[s_i[2]-2:s_i[2]]), int(row_created_at[s_i[2]+1:s_i[3]])).timestamp()
  return dt_i

def efficiency_calc(path='./data/swap.db'):
#NB THIS USES THE CURRENT CONFIG THRESHOLDS IE WHATEVER THEY ARE SET TO NOW, NOT WHAT THEY WERE WHEN DATABASE WAS MADE:
    from config import Config as config_0
    p_retire_lower=config_0().thresholds[0]
    p_retire_upper=config_0().thresholds[1]
    lower_retirement_limit=config_0().lower_retirement_limit
    retirement_limit=config_0().retirement_limit
    user_table=pd.DataFrame.from_dict(read_sqlite(path)['users'])['history']
    subject_histories = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['history']
    subject_golds=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['gold_label']
    
    inefficiency_count=0
    should_be_retired_list=[]
    for i in tqdm(range(len(subject_histories))):
      inefficiency_count_i=0
      history_scores=[]
      for j in range(len(eval(subject_histories[i]))):
        history_scores.append(eval(subject_histories[i])[j][3])
      history_scores=np.array(history_scores)
      inefficiency_count_i_a=0;inefficiency_count_i_b=0;inefficiency_count_i_c=0
#First if: Retirement could only be inefficient if some of the scores are beyond a threshold, and the subject isn't a gold...
#Second if: and only if there are any occurences when the score is beyond the threshold (LHS, below) which is not the last time a classification is made (RHS)
#Third if: and only if two classifications have been made after the subject has been seen more than lower_retirement_limit times:
      if (history_scores<p_retire_lower).any() and subject_golds[i]==-1:
        if np.where(history_scores<p_retire_lower)[0][0]!=len(history_scores)-1:
          if np.sum((history_scores<p_retire_lower)*[1 if i >= lower_retirement_limit else 0 for i in range(len(history_scores))])>=2:
#Think this line below should really be np.sum(np.array([1 if j >= np.where(history_scores<p_retire_lower)[0][0] else 0 for j in range(len(history_scores)))])*[1 if i >= lower_retirement_limit else 0 for i in range(len(history_scores))] >=2 - the current code will underestimate the number of should-be-retired's.
            inefficiency_count_i_a=np.sum((history_scores<p_retire_lower)*[1 if i >= lower_retirement_limit else 0 for i in range(len(history_scores))])-1
#Repeating for the other threshold:
      if (history_scores>p_retire_upper).any() and subject_golds[i]==-1:
        if np.where(history_scores>p_retire_upper)[0][0]!=len(history_scores)-1:
          if np.sum((history_scores>p_retire_upper)*[1 if i >= lower_retirement_limit else 0 for i in range(len(history_scores))])>=2:
            inefficiency_count_i_b=np.sum((history_scores>p_retire_upper)*[1 if i >= lower_retirement_limit else 0 for i in range(len(history_scores))])-1
#Repeating for the majority-vote retirement at N=30 classifications (need the -1 in the len(history_scores) inequality as the history includes the prior:
      if subject_golds[i]==-1 and len(history_scores)-1>retirement_limit:
        inefficiency_count_i_c = (len(history_scores)-1)-retirement_limit
      inefficiency_count+=np.max([inefficiency_count_i_a,inefficiency_count_i_b,inefficiency_count_i_c])
      if np.max([inefficiency_count_i_a,inefficiency_count_i_b,inefficiency_count_i_c])>0:
        should_be_retired_list.append(i)
    subject_ids_which_should_be_retired=(np.array(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id'])[np.array(should_be_retired_list)])
    final_classification_time={}
    penultimate_classification_time={}
#Final classification user is a dictionary of all the subjects, and which users made their final classification.
    final_classification_user={}
    fraction_of_total_user_classifications={}
    delta_time_dict = {}
    for p in should_be_retired_list:
        subj_hist=eval(subject_histories[p])
        subj_id =pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id'][p]
        final_classification_time[subj_id]= (subj_hist[len(subj_hist)-1][5])
        penultimate_classification_time[subj_id] = (subj_hist[len(subj_hist)-2][5])
    for s in tqdm(range(len(user_table))):
        subjects_seen_by_user=[]
        user_history=eval(pd.DataFrame.from_dict(read_sqlite(path)['users'])['history'][s])
        for q in range(len(user_history)):
            subjects_seen_by_user.append(user_history[q][0])
        for r in range(len(subject_ids_which_should_be_retired)):
            if subject_ids_which_should_be_retired[r] in subjects_seen_by_user:
              indx_of_classification=subjects_seen_by_user.index(subject_ids_which_should_be_retired[r])
              if user_history[indx_of_classification][3]==final_classification_time[subject_ids_which_should_be_retired[r]]:
                assert subject_ids_which_should_be_retired[r] not in final_classification_user
                assert pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['retired'][should_be_retired_list[r]]==1
                final_classification_user[subject_ids_which_should_be_retired[r]]=pd.DataFrame.from_dict(read_sqlite(path)['users'])['user_id'][s]
                user_dict_key = 'user_indx: '+str(s)+' total: ' + str(len(subjects_seen_by_user))
                if user_dict_key in fraction_of_total_user_classifications:
                  fraction_of_total_user_classifications[user_dict_key].append(indx_of_classification)
                  delta_time_dict[user_dict_key].append(('i:'+str(indx_of_classification),str(int((date_time_convert(final_classification_time[subject_ids_which_should_be_retired[r]])-date_time_convert(penultimate_classification_time[subject_ids_which_should_be_retired[r]]))/60))+'m'))
                else:
                  fraction_of_total_user_classifications[user_dict_key] = [indx_of_classification]
                  delta_time_dict[user_dict_key]=[('i:'+str(indx_of_classification),str(int((date_time_convert(final_classification_time[subject_ids_which_should_be_retired[r]])-date_time_convert(penultimate_classification_time[subject_ids_which_should_be_retired[r]]))/60))+'m')]
    print('Wasted classifications: ' + str(inefficiency_count))
    print('Total retired so far: '+ str(np.sum(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['retired'])))
#This function looks at the *final* classification made to each subject (and who made it). It looks at the time that final classification was made, and the time of the previous classification.
#It then looks at how many classifications the user had made by the time they made this final classification, as a proportion of the total number of classifications made.
#By looking at consecutive time-steps of the swap.db backups (eg the first few), can show people were shown retired subjects even when there were plenty of others to chose from.
    print('Time (minutes) since previous classification, and number of classifications made so far by that user')
    for i in range(len(delta_time_dict.keys())):
       indx_of_classification_list = []
       user_tuples = delta_time_dict[list(delta_time_dict.keys())[i]]
       tuple_1 = [];tuple_2 = []
       for j in range(len(user_tuples)):
         try:
           tuple_1.append(int(user_tuples[j][0][2:]))
         except:
           print(user_tuples[j][0])
         tuple_2.append(user_tuples[j][1])
       tuple_1_copy = tuple_1.copy()
       #Must only sorting the copy of tuple_1:
       tuple_1_copy.sort()
       sorted_tuples = [('i:' + str(tuple_1_copy[k]),[x for _, x in sorted(zip(tuple_1, tuple_2))][k]) for k in range(len(user_tuples))]
       print(list(delta_time_dict.keys())[i],sorted_tuples)
    return should_be_retired_list
#(efficiency_calc('./data/swap_bFINAL_hardsimsaretest_excludenotloggedon.db'))
#(efficiency_calc('./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon.db'))
#(efficiency_calc('./data/swap_bFINAL_simul_AWS.db'))
#efficiency_calc(path = '/Users/hollowayp/Documents/swap beta25 backup2.db')
#efficiency_calc(path = '/Users/hollowayp/Documents/swap beta25 backup4.db')


def user_class_hist(path='./data/swap.db'):
    user_history=pd.DataFrame.from_dict(read_sqlite(path)['users'])['history']
    N_seen = []
    for i in range(len(user_history)):
        N_seen.append(len(eval(user_history[i])))
    bins = 10**np.linspace(0,np.log10(4*10**4),100)
    N_seen_hist = np.histogram(N_seen,bins=bins)
    bins = N_seen_hist[1]
    bin_mids = [((bins[i]+bins[i+1])/2) for i in range(len(bins)-1)]
    bin_widths = [(bins[i+1]-bins[i]) for i in range(len(bins)-1)]
    N_class_per_bin = bin_mids*N_seen_hist[0]
    fig, ax = pl.subplots(2,figsize=(10,7))
    ax[1].bar(bin_mids,N_class_per_bin,width = bin_widths)
    ax[1].set_xlabel('Number of Classifications')
    ax[1].set_ylabel('Total number of classifications in this bin')
    ax[0].bar(bin_mids,N_seen_hist[0],width = bin_widths)
    ax[0].set_xlabel('Number of Classifications')
    ax[0].set_ylabel('Number of users')
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    pl.show()
    print(N_seen_hist)

#user_class_hist(path='./data/swap_hsc_csv_online.db')
#user_class_hist()

#Function includes *all* training images, not just those of one type or only easy ones. Note this calculates the *fraction* of all images seen which are training (different to other similar function below)
def expected_training_fraction_func(X):
    V = []
    for i in range(len(X)):
      X_i = X[i]
      if X_i<=16:
        V.append((X_i/3)/X_i)
      elif X_i<=16+25:
        V.append(((16/3)+0.2*(X_i-16))/X_i)
      else:
        V.append(((16/3)+(0.2*25)+((X_i-16-25)/10))/X_i)
    return np.array(V)

def random_training_trial():
    training_seen = []
    for i in range(1,500):
        if i<=16:
          training_seen.append(np.random.choice([0,1],p=[2/3,1/3]))
        elif 16<i and i<=16+25:
          training_seen.append(np.random.choice([0,1],p=[4/5,1/5]))
        else:
          training_seen.append(np.random.choice([0,1],p=[9/10,1/10]))
    cumul_freq = np.cumsum(training_seen)/np.arange(1,500)
    return cumul_freq

def gold_frequency(path):
    user_table=pd.DataFrame.from_dict(read_sqlite(path)['users'])['history']
    user_id = pd.DataFrame.from_dict(read_sqlite(path)['users'])['user_id']
    subject_histories = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['history']
    subject_ids=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id']
    subject_golds=np.array(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['gold_label'])
    gold_ids= list(subject_ids[subject_golds!=-1])
#    time_first_classification=[]
#    for s in range(len(user_table)):
#      time_first_classification.append(eval(user_table[s])[1][2])
#    t1 = [date_time_convert(eval(user_table[x])[y][3]) for x in range(len(user_table)) for y in tqdm(range(1,len(eval(user_table[x]))))]
#    t2 = [eval(user_table[x])[y][2] for x in range(len(user_table)) for y in tqdm(range(1,len(eval(user_table[x]))))]
#    pl.scatter((t1-np.min(t1))-(t2-np.min(t2)),t2-np.min(t2))
#    pl.show()
    #Calculating the maximum and minimum clock-times classifications were made (converted from UTC):
    time_array = []
    for x in tqdm(range(0,len(user_table))):
      for y in range(1,len(eval(user_table[x]))):
#        time_array.append(eval(user_table[x])[y][2])
        time_array.append(date_time_convert(eval(user_table[x])[y][3]))
    time_array = np.array(time_array)
    max_time = np.max(time_array)
    min_time = np.min(time_array)
    #time cutoff in seconds (currently 3hr):
    time_cutoff = 3*60*60/(max_time-min_time)
    t_max_below_cutoff = (np.max(time_array[((time_array-min_time)/(max_time-min_time))<time_cutoff])-min_time)/(max_time-min_time)
    print(min_time,max_time, max_time-min_time)
    fig, ax = pl.subplots(1,2,figsize=(14,5))
    #classification cutoff for plot below:
    c_cutoff = 200
    N_less = np.zeros(c_cutoff);N_more = np.zeros(c_cutoff)
    N_many_classifications=0
    x_plot_vals = [];y_plot_vals = [];z_plot_vals = []
    #Iterates making a list of x_i=[classification_number];y_i=[cumulative N. training images];t_i = [classification time] for each user.
    #y_i=[cumulative N. training images] then becomes cumulative training rate.
    for i in tqdm(range(0,len(user_table))):
      x_i = [];y_i = [];t_i = []
#Has to start at 1 as the zeroth index is the prior:
      for j in range(1,len(eval(user_table[i]))):
        if j ==1:
            if eval(user_table[i])[j][0] in gold_ids:
              y_i.append(1)
            else:
              y_i.append(0)
            x_i.append(j)
            t_i.append(date_time_convert(eval(user_table[i])[j][3]))
#            t_i.append(eval(user_table[i])[j][2])
        else:
            if eval(user_table[i])[j][0] in gold_ids:
              y_i.append(y_i[len(y_i)-1]+1)
            else:
              y_i.append(y_i[len(y_i)-1])
            x_i.append(j)
            t_i.append(date_time_convert(eval(user_table[i])[j][3]))
#            t_i.append(eval(user_table[i])[j][2])
      x_i = np.array(x_i)
      y_i = np.array(y_i)/x_i
      t_i = np.array(t_i)
      #Normalise the classification time:
      t_i = (t_i-min_time)/(max_time-min_time)
#      x_plot_vals.extend(list(x_i));y_plot_vals.extend(list(y_i));z_plot_vals.extend(list(t_i))
#      ax[0].plot(x_i,y_i, color='gray',alpha=0.3)
      x_plot_vals.extend(list(x_i[t_i<time_cutoff]));y_plot_vals.extend(list(y_i[t_i<time_cutoff]));z_plot_vals.extend(list(t_i[t_i<time_cutoff]))
      ax[0].plot(x_i[t_i<time_cutoff],y_i[t_i<time_cutoff], color='gray',alpha=0.3)
#      if True:
#        c_cutoff_i = np.min([c_cutoff,len(y_i)])
#        N_many_classifications +=1
#        N_more[0:c_cutoff_i] += ((y_i[0:c_cutoff]>expected_training_fraction_func(np.arange(1,1+c_cutoff_i)))*(t_i[0:c_cutoff]<time_cutoff)).astype('int')
#        N_less[0:c_cutoff_i] += ((y_i[0:c_cutoff]<expected_training_fraction_func(np.arange(1,1+c_cutoff_i)))*(t_i[0:c_cutoff]<time_cutoff)).astype('int')
 #   More_less_ratio = N_more/N_less
    x_plot_vals = np.array(x_plot_vals);y_plot_vals = np.array(y_plot_vals);z_plot_vals = np.array(z_plot_vals)
    norm_0 = colors.Normalize((max_time-min_time)*np.nanmin(z_plot_vals),(max_time-min_time)*np.nanmax(z_plot_vals))
    ax[0].scatter(x_plot_vals[z_plot_vals<time_cutoff],y_plot_vals[z_plot_vals<time_cutoff],c=(max_time-min_time)*z_plot_vals[z_plot_vals<time_cutoff],norm=norm_0,cmap = 'cool',s=15,alpha=0.5)
    #Calculating training rate if follows assigned training frequency:
    for f in range(len(user_table)):
        ax[1].plot(np.arange(1,c_cutoff+1),random_training_trial()[0:c_cutoff],alpha=0.2,c='g')
    sm_0 = pl.cm.ScalarMappable(norm=norm_0, cmap=pl.cm.cool)
    sm_1 = pl.cm.ScalarMappable(norm=pl.Normalize(0, 1), cmap=pl.cm.cool)
    cbar = pl.colorbar(sm_0, ax=ax[0], orientation='vertical')
    cbar.set_label('Time of classification since start/seconds')
    cbar = pl.colorbar(sm_1, ax=ax[1], orientation='vertical')
    for i in range(2):
        ax[i].plot(np.arange(1,800),expected_training_fraction_func(np.arange(1,800)),c='k')
        ax[i].set_xlabel('Number of classifications made by user')
        ax[i].set_ylabel('Running fraction of classifications made which are golds')
        ax[i].set_xlim(0,200)
        ax[i].set_ylim(0,0.8)
    pl.show()
    pl.hist((max_time-min_time)*z_plot_vals/(60*60),bins=100)
    pl.xlabel('Time of classification since start/hours')
    pl.ylabel('Counts')
    pl.show()
    pl.hist((max_time-min_time)*z_plot_vals[z_plot_vals<time_cutoff]/(60),bins=100)
    pl.xlabel('Time of classification since start/minutes')
    pl.ylabel('Counts')
    pl.show()
#    for r in range(N_many_classifications):
#    pl.plot(More_less_ratio,c='b')
#    for t in range(10):
#        N_less_ideal = np.zeros(c_cutoff);N_more_ideal = np.zeros(c_cutoff)
#        for r in range(1000):
        #The oscillations are from the ideal line crossing the thick 'main lines' on the plot (ie the lines where a user doesnt see training images for a long period).
        #This is why the oscillations get slower (the thick lines start off steep then flatten out, hence the ideal line takes longer to cross the later ones)
#            random_trial = random_training_trial()
#            N_more_ideal += (random_trial[0:c_cutoff]>expected_training_fraction_func(np.arange(1,1+c_cutoff))).astype('int')
#            N_less_ideal += (random_trial[0:c_cutoff]<expected_training_fraction_func(np.arange(1,1+c_cutoff))).astype('int')
#        More_less_ratio_ideal = N_more_ideal/N_less_ideal
#        pl.plot(More_less_ratio_ideal,'--',c='k',alpha=0.3)
#    pl.xlabel('Number of classifications made by user')
#    pl.ylabel('Ratio of N. seen over/under set abundance')
#    pl.legend(['M/L beta','M/L ideal'])
#    txt='Only considers classifications made in the first 12 hours (excluding not logged-in)'
#    pl.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
#    pl.show()

#gold_frequency('./data/swap_bFINAL_simul_AWS.db')
#gold_frequency('./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon.db')
#Just looking at first ~12 hours of beta test to see how many golds were shown in early stages. This is also better for colour of scatter points as they are less bunched up near early times (for a long run, most classifications were in the first day, but plotting with uniform time up to say ~7 days).
#GOLD FREQUENCY IS SET DIFFERENTLY FOR NOT-LOGGED-ON USERS (IE JUST SEE NATURAL RATIOS) SO CANT USE THIS:
#gold_frequency('/Users/hollowayp/Documents/swap beta25 backup3.db')

def score_plots(path='./data/swap_bFINAL_hardsimsaretest.db'):
    user_history=pd.DataFrame.from_dict(read_sqlite(path)['users'])['history']
    user_score_lens = [[eval(user_history[i])[j][1]["1"] for j in range(len(eval(user_history[i])))] for i in tqdm(range(len(user_history)))]
    user_score_dud = [[eval(user_history[i])[j][1]["0"] for j in range(len(eval(user_history[i])))] for i in tqdm(range(len(user_history)))]

    #List of lists, [[Change in user-score with each classification] ...for all users]:
    #User Score for identifying lenses:
    delta_user_score_lens = [[(user_score_lens[i][j+1]-user_score_lens[i][j]) for j in range(len(user_score_lens[i])-1)] for i in tqdm(range(len(user_history)))]
    #User Score for identifying duds:
    delta_user_score_dud = [[(user_score_dud[i][j+1]-user_score_dud[i][j]) for j in range(len(user_score_dud[i])-1)] for i in tqdm(range(len(user_history)))]

    fig, ax = pl.subplots(2,3,figsize=(25,10))
    N=[]
    #frac_lens_array gives the fractional change in the (lens) score, for each user, for users who have seen at least 20 images.
    frac_lens_array = []
    for k in tqdm(range(len(delta_user_score_lens))):
      delta_user_score_lens[k] = np.array(delta_user_score_lens[k])
      #Removing occurances where the user score doesn't change (ie don't see a training image of the corresponding type, (i.e. lens in this case)):
      delta_user_score_lens[k] = delta_user_score_lens[k][delta_user_score_lens[k]!=0]
      delta_user_score_lens[k] = [0]+delta_user_score_lens[k]
      N.append(len(delta_user_score_lens[k]))
      ax[0,0].plot(user_score_lens[k])
      ax[0,1].plot(abs(delta_user_score_lens[k]),alpha=0.3)
      #Calculating the fractional change in their score, as a function of the total user-score change, for users who have seen at least 20 training images.
      if len(delta_user_score_lens[k])>=20:
        delta_tot = np.sum(delta_user_score_lens[k])
        frac = [np.sum(delta_user_score_lens[k][0:i]) for i in range(1,len(delta_user_score_lens[k])+1)]
#Adding a '0' so the x axis corresponds to number of training subjects seen.
        frac = [0]+frac
        frac_lens_array.append(np.array(frac)/delta_tot)
        ax[0,2].plot(np.array(frac)/delta_tot,alpha=0.4)
    N=[]
    #frac_lens_array gives the fractional change in the (dud) score, for each user, for users who have seen at least 20 training images.
    frac_dud_array = []
    for k in tqdm(range(len(delta_user_score_dud))):
      delta_user_score_dud[k] = np.array(delta_user_score_dud[k])
      #Removing occurances where the user score doesn't change (ie don't see a training image of the corresponding type, (i.e. duds in this case)):
      delta_user_score_dud[k] = delta_user_score_dud[k][delta_user_score_dud[k]!=0]
      delta_user_score_dud[k] = [0]+delta_user_score_dud[k]
      ax[1,0].plot(user_score_dud[k])
      ax[1,1].plot(abs(delta_user_score_dud[k]),alpha=0.3)
      if len(delta_user_score_dud[k])>=20:
        delta_tot = np.sum(delta_user_score_dud[k])
        frac = [np.sum(delta_user_score_dud[k][0:i]) for i in range(1,len(delta_user_score_dud[k])+1)]
        frac = [0]+frac
        frac_dud_array.append(np.array(frac)/delta_tot)
        ax[1,2].plot(np.array(frac)/delta_tot,alpha=0.4)
    def average_delta_user_score_func(delta_user_score,error=False):
        average_delta_user_score = []
        error_bar_lower = []
        error_bar_upper = []
        #Only calculating for first 50 classifications of training images (of each type):
        #List of lists, [[Change in user score for all users for i'th training image] ...for all training image classifications]:
        for i in tqdm(range(50)):
          average_delta_user_score.append([])
          for j in range(len(delta_user_score)):
            try:
              average_delta_user_score[i].append((delta_user_score[j][i]))
            except IndexError:
              continue
          average_delta_user_score[i] = np.array(average_delta_user_score[i])
          median = np.median(average_delta_user_score[i])
          error_bar_lower.append(np.std(average_delta_user_score[i]))
          error_bar_upper.append(np.std(average_delta_user_score[i]))
          #Now replace with the median score for the i'th classification:
          if error ==False:
            average_delta_user_score[i] = np.median(abs(average_delta_user_score[i]))
          else:
            average_delta_user_score[i] = np.median((average_delta_user_score[i]))
        if error == False:
            return np.array(average_delta_user_score)
        else:
            return np.array(average_delta_user_score), error_bar_lower,error_bar_upper
    average_frac_lens,error_lens_lower,error_lens_upper = average_delta_user_score_func(frac_lens_array,error=True)
    print(error_lens_lower)
    average_frac_dud, error_dud_lower,error_dud_upper = average_delta_user_score_func(frac_dud_array,error=True)
    print(error_dud_lower)
    ax[0,2].plot(average_frac_lens,'-.',color='k')
    #Find the average fractional change of user score for lenses and duds combined, for users who have seen >20 training images.
    print('Numbers to note:')
    #Input list is [[fraction_of_total_dud_score] for all users] concatenated with [[fraction_of_total_lens_score] for all users]
    #=> Producing [[fraction_of_total_dud_score] for all users,[fraction_of_total_lens_score] for all users] as required.
    print(average_delta_user_score_func(frac_dud_array+frac_lens_array,error=True)[0])
    ax[1,2].plot(average_frac_dud,'-.',color='k')
    ax[0,2].errorbar(np.arange(len(average_frac_lens)),average_frac_lens,yerr = [error_lens_lower,error_lens_upper],color='k')
    ax[1,2].errorbar(np.arange(len(average_frac_dud)),average_frac_dud,yerr = [error_dud_lower,error_dud_upper],color='k')
    ax[0,2].set_xlabel('Number of (Lens) training subjects seen')
    ax[1,2].set_xlabel('Number of (Duds) training subjects seen')
    ax[0,2].set_ylabel('User score/Final User Score')
    ax[1,2].set_xlabel('User score/Final User Score')
    ax[0,2].set_title('Lens')
    ax[1,2].set_title('Duds')
    ax[0,2].set_ylim(-1,1.8)
    ax[1,2].set_ylim(-1,1.8)
    average_delta_user_score_lens = average_delta_user_score_func(delta_user_score_lens)
    average_delta_user_score_dud = average_delta_user_score_func(delta_user_score_dud)
    a =average_delta_user_score_lens.copy();b =average_delta_user_score_dud.copy()
    ax[0,1].set_xlim(1,len(a[np.array(1-np.isnan(a)).astype('bool')])+1)
    ax[1,1].set_xlim(1,len(b[np.array(1-np.isnan(b)).astype('bool')])+1)
    ax[0,1].plot(average_delta_user_score_lens,'-.',color='k')
    ax[1,1].plot(average_delta_user_score_dud,'-.',color='k')
    ax[0,1].plot(average_delta_user_score_lens,'-.',color='k')
    ax[1,1].plot(average_delta_user_score_dud,'-.',color='k')
#Adding new axis 'number of golds seen PER TYPE OF GOLDS IE DIVIDED BY 2' based on frequency of occurence in beta test (1 in 3 for first 16, then 1 in 5 for next 25, then 1 in 10).
    ax2 = ax[0,0].twiny()
    ax2.set_xlabel('Average number of (Lens) training subjects seen')
    ax3 = ax[1,0].twiny()
    ax3.set_xlabel('Average number of (Dud) training subjects seen')
    new_tick_locations = np.arange(0,1000,100)
    def tick_function(X):
        V = []
        for i in range(len(X)):
          X_i = X[i]
          if X_i<=16:
            V.append(X_i/3)
          elif X_i<=16+25:
            V.append((16/3)+0.2*(X_i-16))
          else:
            V.append((16/3)+(0.2*25)+((X_i-16-25)/10))
#MULTIPLYING BY (82/102) HERE AS PLOTTING FROM A HARDSIMSARETEST DATABASE SO THERE ARE ONLY 82 NOT 102 TRAINING SUBJECTS...
#...NEED TO REMOVE [AND SWAP WITH  V = np.round(V)/2] IF HARD SIMS INCLUDED AS TRAINING
        V = (82/102)*np.round(V)/2
        return ["%d" % z for z in V]
    ax2.set_xlim(ax[0,0].get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax3.set_xlim(ax[0,0].get_xlim())
    ax3.set_xticks(new_tick_locations)
    ax3.set_xticklabels(tick_function(new_tick_locations))
    ax[0,0].set_title('Lens')
    ax[1,0].set_title('Dud')
    ax[0,1].set_title('Lens')
    ax[1,1].set_title('Dud')
    #Finding the user with the longest classification history:
    f_i = 0;f_max = 0
    for f in tqdm(range(len(user_history))):
      if len(eval(user_history[f]))>f_max:
        f_max = len(eval(user_history[f]))
        f_i = f
    ax[0,0].plot(user_score_lens[f_i],color='k')
    ax[0,1].plot(abs(delta_user_score_lens[f_i]),color='k',alpha=0.7)
    ax[1,0].plot(user_score_dud[f_i],color='k')
    ax[1,1].plot(abs(delta_user_score_dud[f_i]),color='k',alpha=0.7)
    for p in range(2):
        ax[p,0].set_ylabel('User score')
        ax[p,0].set_xlabel('Number of subjects seen')
        ax[p,1].set_ylabel('|Change in user score|')
    ax[0,1].set_xlabel('Number of (Lens) training subjects seen')
    ax[1,1].set_xlabel('Number of (Dud) training subjects seen')
    pl.show()

#score_plots()

def scatter_hist(x, y, s, ax, ax_histx, ax_histy,not_logged_on=False):
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    y_bar = [];x_bar = []
    x = np.array(x);y=np.array(y);s=np.array(s)
    bins = np.arange(-0.025, 1.075, 0.05)
    bar_mids = np.array([(bins[j]+bins[j+1])/2 for j in range(len(bins)-1)])
    for i in range(len(bins)-1):
      x_bar.append(np.sum((x[(bins[i]<=x)&(x<bins[i+1])]*s[(bins[i]<=x)&(x<bins[i+1])])))
      y_bar.append(np.sum((y[(bins[i]<=y)&(y<bins[i+1])]*s[(bins[i]<=y)&(y<bins[i+1])])))
    print(x_bar,y_bar)
    if not_logged_on==True:
        ax.scatter(x, y,s=(1+s)/15,c='red',marker='x')
        ax_histx.bar(bar_mids,x_bar,width = 0.05,alpha=0.5,color = 'red')
        ax_histy.barh(bar_mids,width = y_bar,height = 0.05,alpha=0.5,color='red')
#        ax_histx.hist(x, bins=bins,alpha=0.5,color='red')
#        ax_histy.hist(y, bins=bins, orientation='horizontal',alpha=0.5,color='red')
    else:
        ax.scatter(x, y,s=(1+s)/15,c='blue')
        ax_histx.bar(bar_mids,x_bar,width = 0.05,alpha=0.5,color = 'blue')
        ax_histy.barh(bar_mids,width=y_bar,height = 0.05,alpha=0.5,color='blue')
#        ax_histx.hist(x, bins=bins,alpha=0.5,color='blue')
#        ax_histy.hist(y, bins=bins, orientation='horizontal',alpha=0.5,color='blue')
    ax.set_xlabel('P("LENS"|LENS)')
    ax.set_ylabel('P("NOT"|NOT)')
    ax_histx.set_xlim(0,1)
    ax_histy.set_ylim(0,1)


hard_sims_ids=[72109491, 72109492, 72109493, 72109494, 72109495, 72109496, 72109497, 72109498, 72109499, 72109500, 72109501, 72109502, 72109503, 72109504, 72109505, 72109506, 72109507, 72109508, 72109509, 72109510]
#FILE PATH HERE
def trajectory_plot(path='./data/swap.db', subjects=[]):
#    print(pd.DataFrame.from_dict(read_sqlite(path)['subjects']))
    subject_id=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id']
    subject_histories = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['history']
    histories_df = pd.DataFrame(subject_histories)
    subject_golds=list(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['gold_label'])
    N_class = 0; N_class_hist = [];N_subj = 0
    for i in tqdm(range(len(subject_id))):
      subj_i = eval(subject_histories[i])
      if subject_golds[i]==-1 and subj_i[len(subj_i)-1][3]<10**-5:
        N_subj+=1
        N_class+=(len(eval(subject_histories[i]))-1)
        N_class_hist.append((len(eval(subject_histories[i]))-1))
    print(N_subj, N_class,N_class/800)
    print(np.median(N_class_hist))
#    pl.hist(N_class_hist,bins=20)
#    pl.show()
    retired=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['retired']
    retired = sum(1 for x in retired if x!=0)
    users =pd.DataFrame.from_dict(read_sqlite(path)['users'])
    user_score_matrices=users['user_score']
    user_confusion_matrices=users['confusion_matrix']
    hard_sims_ids_indx=[]
    for f in tqdm(range(len(subject_id))):
      if subject_id[f] in hard_sims_ids:
        hard_sims_ids_indx.append(f)
    if len(subject_id)!=len(set(subject_id)):
      print('WARNING: DUPLICATION OF SUBJECTS WITHIN DB')
    hist_i=[]
    for i in range(len(histories_df)):
      hist_i.append(len(eval(histories_df['history'][i])))
    # get subjects
    # max_seen is set by subject with max number of classifications
    max_seen = 1
    subjects=len(set(subject_id))

    print('Plotting from ' + str(subjects) + ' subjects of which ' + str(retired) + ' are retired and '  + str(len(histories_df.loc[histories_df['history']!='[["_", "_", "_", 0.0005, "_", "_"]]'])) + ' are (uniquely) classified')
    subjects_final = []
    subjects_history_final=[]
    history_all_final=[]
    golds_final=[]
    classification_time_final=[]
    for indx in tqdm(range(len(subject_id))):
        subject = subject_id[indx]
        subjects_final.append(subject)
        golds_final.append(subject_golds[indx])
        history_i=[]
        classification_time_i=[]
        history_all_i=[]
        history_list_i=eval(subject_histories[indx])
        for i in range(len(history_list_i)):
          history_i.append(history_list_i[i][3])
          date_i=history_list_i[i][5]
          if i==0:
            classification_time_i.append(-100)
          else:
            classification_time_i.append(datetime.datetime(int(date_i[0:4]),int(date_i[5:7]),int(date_i[8:10]),int(date_i[11:13]),int(date_i[14:16]),int(date_i[17:19])).timestamp())
          history_all_i.append(history_list_i[i])
        subjects_history_final.append(history_i)
        history_all_final.append(history_all_i)
        if len(classification_time_i)!=1:
            classification_time_i=np.array(classification_time_i)-classification_time_i[1]
        classification_time_i=np.array(classification_time_i)
        classification_time_i[0]=-100
        classification_time_final.append(classification_time_i)
        if len(history_i) > max_seen:
            max_seen = len(history_i)-1

    subjects = subjects_final
    p_real , p_bogus= thresholds_setting()

    def plotting_traj(subjects, subjects_history_final,timer=False):
        from config import Config as config_0
        fig, ax = pl.subplots(figsize=(3,3), dpi=300)
        ax.tick_params(labelsize=5)

        color_test = 'gray';color_bogus = 'red';color_real = 'blue'
        colors = [color_bogus, color_real, color_test]
        
        linewidth_test = 1.0; linewidth_bogus = 1.5; linewidth_real = 1.5
        linewidths = [linewidth_bogus, linewidth_real, linewidth_test]
        
        alpha_test = 0.1; alpha_bogus = 0.3; alpha_real = 0.3
        alphas = [alpha_bogus, alpha_real, alpha_test]

        # axes and labels
        p_min = 10**-10
        p_max = 1
        if timer==False:
            ax.set_xlim(p_min, p_max)
            ax.set_xscale('log')
            ax.set_ylim(max_seen+1,0)
#            ax.set_ylim(31,0)
            ax.set_xlim(10**-10,1)
            ax.axvline(x=prior_setting(), color=color_test, linestyle='dotted')
            ax.axvline(x=p_bogus, color=color_bogus, linestyle='dotted')
            ax.axvline(x=p_real, color=color_real, linestyle='dotted')
            ax.fill_betweenx(x1=p_min,x2=p_bogus,y=[max_seen+1,config_0().lower_retirement_limit],alpha=0.2,color='darkviolet')
            ax.fill_betweenx(x1=p_bogus,x2=p_max,y=[max_seen+1,config_0().retirement_limit],alpha=0.2,color='darkviolet')
            ax.set_xlabel('Posterior Probability Pr(LENS|d)',fontsize=5)
            ax.set_ylabel('Number of Classifications',fontsize=5)
            sub_list=[]
            print(str(len(should_be_retired_list)) + ' subjects should have been retired but were subsequently classified')
            for j in range(0,len(subjects)):
                if j in np.linspace(0,len(subjects)):
                  print(j)
                history = np.array(subjects_history_final[j])
                y = np.arange(len(history) + 1)-1
                history = np.append(prior_setting(), history)
#                if j not in should_be_retired_list:
#                    ax.plot(history, y, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],linewidth=0.5)
#                    ax.scatter(history, y,marker='+',linewidth=1,color=colors[golds_final[j]],alpha=0.6,s=0.5)
#                    ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,edgecolors=colors[golds_final[j]],facecolors=colors[golds_final[j]])
                if j in should_be_retired_list:
                  ax.plot(history, y, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],linewidth=0.5)
                  ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,c='k')
                  abc=[]
                  for s in range(len(classification_time_final[j])):
                    abc.append(str(datetime.timedelta(seconds=classification_time_final[j][s])))
#                if j in hard_sims_ids_indx:
#                    ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,c='g')
#                    ax.plot(history, y, linestyle='-',alpha=0.3,color='g',linewidth=0.5)
#            colors.append('g');alphas.append(0.3)
            patches = []
#            for color, alpha, label in zip(colors, alphas, ['Training: Dud', 'Training: Easy sim', 'Test','Training: Hard sim']):
            for color, alpha, label in zip(colors, alphas, ['Training: Dud', 'Training: Lens', 'Test']):
                patch = mpatches.Patch(color=color, alpha=alpha, label=label)
                patches.append(patch)
            ax.legend(handles=patches, loc='lower right', framealpha=1.0,prop={'size':5})
            fig.tight_layout()
            pl.show()
        else:
            ax.set_xlim(p_min, p_max)
            ax.set_xscale('log')
            ax.set_ylim(bottom=1,top=1.1*np.max([item for sublist in classification_time_final for item in sublist]))
            ax.axvline(x=prior_setting(), color=color_test, linestyle='dotted')
            ax.axvline(x=p_bogus, color=color_bogus, linestyle='dotted')
            ax.axvline(x=p_real, color=color_real, linestyle='dotted')
            ax.set_xlabel('Posterior Probability Pr(LENS|d)',fontsize=5)
            ax.set_ylabel('Time since first classification/s',fontsize=5)
            sub_list=[]
#            ax.set_yscale('log')
            #NB Truncating Here:
            for j in range(0,len(subjects)):
                printer(j,len(subjects))
                history = np.array(subjects_history_final[j])
                y = classification_time_final[j]
                ax.plot(history[1:len(history)], y[1:len(y)]+1, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],linewidth=0.5)
#                for f in range(1,len(y)):
#                  ax.annotate(datetime.timedelta(seconds=y[f]), (history[f], y[f]),fontsize=3)
                ax.scatter(history, y,marker='+',linewidth=1,color=colors[golds_final[j]],alpha=1,s=1)
                ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,edgecolors=colors[golds_final[j]],facecolors=colors[golds_final[j]])
                if j in should_be_retired_list:
                  ax.scatter(history, y,marker='+',linewidth=1,color='k',s=1)
                  ax.scatter(history[-1:], y[-1:], s=1,color='k')
                  ax.plot(history[1:len(history)], y[1:len(y)]+1, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],linewidth=0.5)
                  for f in range(1,len(y)):
                    ax.annotate(datetime.timedelta(seconds=y[f]), (history[f], y[f]),fontsize=3)
            patches = []
            for color, alpha, label in zip(colors, alphas, ['Bogus', 'Real', 'Test']):
                patch = mpatches.Patch(color=color, alpha=alpha, label=label)
                patches.append(patch)
            for t in range(1,10,2):
                ax.annotate(str(t)+'minutes',(p_min,60*t),fontsize=3)
                ax.axhline(y=60*t, color='k',linewidth=0.05)
                ax.annotate(str(t)+'hours',(p_min,60*60*t),fontsize=3)
                ax.axhline(y=60*60*t, color='k',linewidth=0.05)
                ax.annotate(str(t)+'days',(p_min,24*60*60*t),fontsize=3)
                ax.axhline(y=24*60*60*t, color='k',linewidth=0.05)
            ax.legend(handles=patches, loc='lower right', framealpha=1.0,prop={'size':5})
            pl.show()

    plotting_traj(subjects, subjects_history_final,timer=False)
    
    posterior_prob_final=[]
    for i in range(len(subjects_history_final)):
      posterior_prob_final.append(subjects_history_final[i][len(subjects_history_final[i])-1])
    N_class_final=[]
    for i in range(len(history_all_final)):
#Need to subtract 1 (as below) as unclassified subjects still have the prior in their history.
      N_class_final.append(len(history_all_final[i])-1)

    def plotting_hist(N_class_final,subjects_history_final):
      N_class_bins=[[],[],[]]
      for k in range(len(N_class_final)):
        N_class_bins[golds_final[k]+1].append(N_class_final[k])
      edgecolors=['grey','red','blue']
      class_name=['Test','Dud','Lens']
      for r in range(3):
        print(class_name[r]+': '+ str(np.histogram(N_class_bins[r],bins=(max(N_class_final)-min(N_class_final))+1,range=(min(N_class_final),max(N_class_final)+1))[0]))
        pl.hist(N_class_bins[r],bins=(max(N_class_final)-min(N_class_final))+1,range=(min(N_class_final),max(N_class_final)+1),                      stacked=False,edgecolor=edgecolors[r],fill=False,align='left',density=True)
      pl.xlabel('Number of Classifications')
      pl.ylabel('N')
      pl.yscale('log')
      pl.title(path)
      pl.legend(['Test','Dud','Lens'])
      pl.show()
      post_prob_bins=[[],[],[]]
      for j in range(len(posterior_prob_final)):
        if len(subjects_history_final[j])!=1:
          post_prob_bins[golds_final[j]+1].append(posterior_prob_final[j])
      print(post_prob_bins[1])
      pl.hist(post_prob_bins[0],bins=np.logspace(np.log10(1e-20),np.log10(1.0), 50),stacked=False,edgecolor = 'grey',fill=False)
      pl.hist(post_prob_bins[1],bins=np.logspace(np.log10(1e-20),np.log10(1.0), 50),stacked=False,edgecolor='red',fill=False)
      pl.hist(post_prob_bins[2],bins=np.logspace(np.log10(1e-20),np.log10(1.0), 50),stacked=False,edgecolor='blue',fill=False)
      pl.legend(['Test','Training: Dud','Training: Lens'])
      pl.xlabel('Posterior Probability')
      pl.ylabel('N')
      pl.title(path)
#      pl.yscale('log')
      pl.xscale('log')
      pl.show()
#    plotting_hist(N_class_final,subjects_history_final)

    def retrieve_list(list_path):
        try:
            df = pd.read_csv(list_path)
            list_total = []
            for row in df.iterrows():
                try:
                  list_i = eval(row[1]['list items'])
                except:
                  list_i=row[1]['list items']
                list_total.append(list_i)
            return list_total
        except FileNotFoundError:
            return []

#FILE PATH HERE
#    df_s=pd.DataFrame.from_dict(read_sqlite('./data/swap.db')['subjects'])

    def perc_plot(gold_label_i):
        assert gold_label_i in (0,1)
        if gold_label_i==1:
              p_level=5
        if gold_label_i==0:
              p_level=95
        subjects_checked={}
        subject_score_dict={}
        percentile_list=[]
        N = len(tuples_list)
        for q in range(N):
          printer(q,N)
          subject_q=tuples_list[q][0]
          if int(df_s.loc[df_s['subject_id'] == subject_q]['gold_label'])==gold_label_i:
              try:
                n = subjects_checked[subject_q]
                subjects_checked[subject_q]+=1
                subject_score_dict[subject_q]=eval(list(df_s.loc[df_s['subject_id'] == subject_q]['history'])[0])[n][3]
              except:
                subjects_checked[subject_q]=1
                subject_score_dict[subject_q]=eval(list(df_s.loc[df_s['subject_id'] == subject_q]['history'])[0])[0][3]
          if q%100==0 and q!=0:
            percentile_list.append(np.percentile(list(subject_score_dict.values()),p_level))

        pl.plot(100*np.arange(len(percentile_list)),percentile_list)
        pl.xlabel('N. Classifications')
        pl.ylabel('P_gold= '+str(gold_label_i))
        pl.show()

#    perc_plot(0)
#    perc_plot(1)
#Offline
#should_be_retired_list= efficiency_calc('./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon_offlineswap.db')
#trajectory_plot('./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon_offlineswap.db')
#Online, test
#should_be_retired_list= efficiency_calc('./data/swap_bFINAL_hardsimsaretest_excludenotloggedon.db')
#should_be_retired_list=[]
#trajectory_plot('./data/swap_bFINAL_hardsimsaretest_excludenotloggedon.db')
#trajectory_plot('./data/swap_bFINAL_hardsimsaretest.db')
#Online, training
#should_be_retired_list= efficiency_calc('./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon.db')
#trajectory_plot('./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon.db')
#trajectory_plot('./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon.db')
#trajectory_plot('./data/swap_bFINAL_hardsimsaretraining.db')
#AWS swap
#should_be_retired_list= efficiency_calc('./data/swap_bFINAL_simul_AWS.db')
#should_be_retired_list=[]
#trajectory_plot('./data/swap_bFINAL_simul_AWS.db')
#trajectory_plot('/Users/hollowayp/Documents/swap beta25 backup2.db')
#trajectory_plot('./data/swap_hsc_csv_online.db')
#should_be_retired_list=efficiency_calc('/Users/hollowayp/Documents/swap beta25 backup2.db')
#trajectory_plot('/Users/hollowayp/Documents/swap beta25 backup2.db')
#should_be_retired_list=efficiency_calc('/Users/hollowayp/Documents/swap beta25 backup4.db')
#trajectory_plot('/Users/hollowayp/Documents/swap beta25 backup4.db')
#trajectory_plot('./data/swap_hsc_online_includeloggedon.db')
#trajectory_plot('./data/swap_hsc_online_excludeloggedon.db')

def plotting_user_score(path):
  users =pd.DataFrame.from_dict(read_sqlite(path)['users'])
  user_score_matrices=users['user_score']
  user_confusion_matrices=users['confusion_matrix']
  left, width = 0.1, 0.65
  bottom, height = 0.1, 0.65
  spacing = 0.005
  rect_scatter = [left, bottom, width, height]
  rect_histx = [left, bottom + height + spacing, width, 0.2]
  rect_histy = [left + width + spacing, bottom, 0.2, height]
  # start with a square Figure
  fig = pl.figure(figsize=(8, 8))
  ax = fig.add_axes(rect_scatter)
  ax_histx = fig.add_axes(rect_histx, sharex=ax)
  ax_histy = fig.add_axes(rect_histy, sharey=ax)
  N=0;Tot=0

  def f(p):
     try:
         if pd.DataFrame.from_dict(read_sqlite(path)['users'])['user_id'][p][0:13]=='not-logged-in':
           return 0
         else:
           return 1
     except:
       return 1
  print('Starting multiprocessing')
  st = time.time()
  #with mp.Pool() as pool:
  #    a = pool.map(f, np.arange(len(user_score_matrices)))
  #print(len(a),np.sum(a))
  et = time.time()
  print(et-st)
  print('Now running user score plot code')
  x_NL=[];x_L = [];y_NL = [];y_L=[];s_L = [];s_NL=[]
#      for i in range(len(user_score_matrices)):
#        p = np.random.randint(0,len(user_score_matrices))
  for p in tqdm(range(len(user_score_matrices))):
    try:
        if pd.DataFrame.from_dict(read_sqlite(path)['users'])['user_id'][p][0:13]=='not-logged-in':
          x_NL.append(eval(user_score_matrices[p])['1']);y_NL.append(eval(user_score_matrices[p])['0']);s_NL.append(sum(eval(user_confusion_matrices[p])['n_seen']))
#              pl.scatter(eval(user_score_matrices[p])['1'],eval(user_score_matrices[p])['0'],s=(1+sum(eval(user_confusion_matrices[p])['n_seen']))/15,color='r',marker='x')
        else:
          x_L.append(eval(user_score_matrices[p])['1']);y_L.append(eval(user_score_matrices[p])['0']);s_L.append(sum(eval(user_confusion_matrices[p])['n_seen']))
#              pl.scatter(eval(user_score_matrices[p])['1'],eval(user_score_matrices[p])['0'],s=(1+sum(eval(user_confusion_matrices[p])['n_seen']))/15,color='blue')
    except:
        x_L.append(eval(user_score_matrices[p])['1']);y_L.append(eval(user_score_matrices[p])['0']);s_L.append(sum(eval(user_confusion_matrices[p])['n_seen']))
#            pl.scatter(eval(user_score_matrices[p])['1'],eval(user_score_matrices[p])['0'],s=(1+sum(eval(user_confusion_matrices[p])['n_seen']))/15,color='blue')
  x_NL=np.array(x_NL);x_L = np.array(x_L);y_NL = np.array(y_NL);y_L=np.array(y_L);s_L = np.array(s_L);s_NL=np.array(s_NL)
  scatter_hist(x_NL, y_NL, s_NL, ax, ax_histx, ax_histy, not_logged_on=True)
  scatter_hist(x_L, y_L, s_L, ax, ax_histx, ax_histy)
#      pl.gca().set_aspect('equal')
  pl.show()
  fig, ax = pl.subplots(2,figsize=(10,7))
  ax[0].hist2d(x_NL,y_NL,bins=21,density=True,range=[[-0.025,1.025],[-0.025,1.025]],norm=mpl.colors.LogNorm())
  ax[1].hist2d(x_L,y_L,bins=21,density=True,range=[[-0.025,1.025],[-0.025,1.025]],norm=mpl.colors.LogNorm())
  pl.show()
  
#plotting_user_score('./data/swap_hsc_online_excludeloggedon.db')


def compare_swap_methods(db_path_1,db_path_2):
    subject_scores1 = np.array(pd.DataFrame.from_dict(read_sqlite(db_path_1)['subjects'])['score']).astype('float32')
    subject_scores2 = np.array(pd.DataFrame.from_dict(read_sqlite(db_path_2)['subjects'])['score']).astype('float32')
    subject_seen1 = np.array(pd.DataFrame.from_dict(read_sqlite(db_path_1)['subjects'])['seen']).astype('int')
    subject_seen2 = np.array(pd.DataFrame.from_dict(read_sqlite(db_path_2)['subjects'])['seen']).astype('int')
    print(np.sum(subject_seen1),np.sum(subject_seen2))
    pl.hist(np.log10(subject_scores1)-np.log10(subject_scores2),bins=40)
    pl.xlabel('log10(Score_1)-log10(Score_2)')
    pl.yscale('log')
    pl.show()
    
#compare_swap_methods('./data/swap_bFINAL_hardsimsaretest.db','./data/swap_bFINAL_hardsimsaretest_excludenotloggedon.db')
#compare_swap_methods('./data/swap_bFINAL_simul_AWS.db','./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon.db')
#compare_swap_methods('./data/swap_bFINAL_simul_AWS.db','./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon_changingalreadyseen3tofalse.db')

#Double check the databases are correct (ie made correctly!)
def compare_logged_off(db_path_1,db_path_2):
    print(pd.DataFrame.from_dict(read_sqlite(db_path_1)['users']))
    subject_seen1 = np.array(pd.DataFrame.from_dict(read_sqlite(db_path_1)['subjects'])['seen']).astype('int')
    subject_seen2 = np.array(pd.DataFrame.from_dict(read_sqlite(db_path_2)['subjects'])['seen']).astype('int')
    subject_scores1 = np.array(pd.DataFrame.from_dict(read_sqlite(db_path_1)['subjects'])['score']).astype('float32')
    subject_scores2 = np.array(pd.DataFrame.from_dict(read_sqlite(db_path_2)['subjects'])['score']).astype('float32')
    subject_scores1_scaled = (np.log10(subject_scores1) - np.log10(prior_setting()))*(subject_seen2/subject_seen1)
    subject_scores2_scaled = (np.log10(subject_scores2) - np.log10(prior_setting()))
#    pl.hist(subject_scores1_scaled-subject_scores2_scaled,bins=40)
#    pl.yscale('log')
#    pl.show()
    subject_id1=np.array(pd.DataFrame.from_dict(read_sqlite(db_path_1)['subjects'])['subject_id']).astype('int')
    subject_id2=np.array(pd.DataFrame.from_dict(read_sqlite(db_path_2)['subjects'])['subject_id']).astype('int')
    for i in range(len(subject_id1)):
      if subject_id1[i]!=subject_id2[i]:
        print('Subject ids dont match!')
    #Now cropping the full database histories to the length they would be had they excluded not-logged-on's.
    subject_id=pd.DataFrame.from_dict(read_sqlite(db_path_1)['subjects'])['subject_id']
    subject_histories = pd.DataFrame.from_dict(read_sqlite(db_path_1)['subjects'])['history']
    histories_df = pd.DataFrame(subject_histories)
    subject_golds=list(pd.DataFrame.from_dict(read_sqlite(db_path_1)['subjects'])['gold_label'])
    diff_hist = []
    for indx in range(len(subject_id)):
        history_i=eval(subject_histories[indx])
        history_i = [history_i[j][3] for j in range(len(history_i))]
        history_list_i=eval(subject_histories[indx])
        for i in range(len(history_list_i)):
          history_i.append(history_list_i[i][3])
        cropped_score = history_i[subject_seen2[indx]-1]
        if subject_golds[indx]==-1:
          diff_hist.append(np.log10(cropped_score)-np.log10(subject_scores2[indx]))
    pl.hist(diff_hist,bins=20)
    pl.show()
#compare_logged_off('./data/swap_bFINAL_hardsimsaretest.db','./data/swap_bFINAL_hardsimsaretest_excludenotloggedon.db')

def identified_classes(path='./data/swap_bFINAL_simul_AWS.db',retired_as = 1):
    subject_retired_status = np.array(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['retired_as'])
    subject_id = np.array(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id'])
    subject_indx = [i for i in range(len(subject_retired_status)) if (subject_retired_status[i]==retired_as or subject_retired_status[i]==str(retired_as))]
    return subject_id[subject_indx]

#identified_lenses()

def neural_net_compare():
#    lens_candidates = identified_classes(retired_as = 1)
#    dud_candidates = identified_classes(retired_as = 0)
    subject_id_set=[]
    metadata_set=[]
    with open('/Users/hollowayp/Downloads/space-warps-des-subjects-11.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for i,row in enumerate(reader):
            if row['workflow_id']=='8878':
                subject_id_set.append(row['subject_id'])
                metadata_set.append(row['metadata'])
    test_subject_id = [];rm_score_set=[];real_score_set = []
    for i in range(len(metadata_set)):
      try:
        rm_score_set.append(float(eval(metadata_set[i])['#rm_score']))
        real_score_set.append(float(eval(metadata_set[i])['#real_score']))
        test_subject_id.append(int(subject_id_set[i]))
      except:
        continue
    pl.hist(rm_score_set,bins=20,alpha=0.5)
    pl.hist(real_score_set,bins=20,alpha=0.5)
    pl.show()
    assert len(rm_score_set)==len(test_subject_id)
    subject_gold_SW_tot = list(pd.DataFrame.from_dict(read_sqlite('./data/swap_bFINAL_simul_AWS.db')['subjects'])['gold_label'])
    subject_id_SW_tot = list(pd.DataFrame.from_dict(read_sqlite('./data/swap_bFINAL_simul_AWS.db')['subjects'])['subject_id'])
    subject_scores_SW_tot = list(pd.DataFrame.from_dict(read_sqlite('./data/swap_bFINAL_simul_AWS.db')['subjects'])['score'])
    test_subject_scores = []
    for k in range(len(test_subject_id)):
      indx3 = subject_id_SW_tot.index(test_subject_id[k])
      assert subject_id_SW_tot[indx3] == test_subject_id[k]
#All subjects in test_subject_id should be test subjects as the golds (or at least the simulations) shouldn't have scores from the neural network? Next line checks this:
      assert subject_gold_SW_tot[indx3] ==-1
      test_subject_scores.append(float(subject_scores_SW_tot[indx3]))
    NNY_SWY = [];NNN_SWY = [];NNY_SWN = [];NNN_SWN=[]; NN_tot_P = 0; NN_SW_P = 0
    for m in range(len(test_subject_id)):
      NN_score_i = float(rm_score_set[m])
      if NN_score_i>0.65:
        NN_tot_P+=1
        if test_subject_scores[m]>10**-5:
          NN_SW_P+=1
      if float(test_subject_scores[m])>10**-1 and NN_score_i>0.9 and len(NNY_SWY)<5:
        NNY_SWY.append(test_subject_id[m])
      if float(test_subject_scores[m])>10**-2 and NN_score_i<0.1 and len(NNN_SWY)<5:
        NNN_SWY.append(test_subject_id[m])
      if float(test_subject_scores[m])<10**-5 and NN_score_i>0.9 and len(NNY_SWN)<5:
        NNY_SWN.append(test_subject_id[m])
      if float(test_subject_scores[m])<10**-5 and NN_score_i<0.1 and len(NNN_SWN)<5:
        NNN_SWN.append(test_subject_id[m])
    print(NN_tot_P,NN_SW_P)
    print(NNY_SWY)
    print(NNN_SWY)
    print(NNY_SWN)
    print(NNN_SWN)
    print(len(test_subject_scores),len(rm_score_set))
    for i in range(len(test_subject_scores)):
      if test_subject_scores[i]<10**-5:
        pl.scatter(test_subject_scores[i],rm_score_set[i],c='r')
      else:
        pl.scatter(test_subject_scores[i],rm_score_set[i],c='b')
    pl.ylabel('NN Score')
    pl.xlabel('SW Score')
    pl.xscale('log')
    pl.show()

#neural_net_compare()

def majority_vote(sequence):
  occurence_count = Counter(sequence)
  return occurence_count.most_common(1)[0][0]

def test_maj_vote():
    i = [105,461,129,359]
    for k in i:
        subject_history = pd.DataFrame.from_dict(read_sqlite('./data/swap_bFINAL_simul_AWS.db')['subjects'])['history']
        subject_history=eval(subject_history[k])
        print([int(h[2]) for h in subject_history[1:len(subject_history)]])
        print(len([int(h[2]) for h in subject_history[1:len(subject_history)]]))
        print(sum([int(h[2]) for h in subject_history[1:lwen(subject_history)]]))
        print(majority_vote([h[2] for h in subject_history[1:len(subject_history)]]))

#test_maj_vote()

def show_databases(path):
    users=pd.DataFrame.from_dict(read_sqlite(path)['users'])
    subjects = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])
    subject_golds=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])
    print(users)

#users=pd.DataFrame.from_dict(read_sqlite('/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/swap_test_db2.db')['users'])
#show_databases('/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/swap_test_db2.db')
#show_databases('/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/swap_hsc_online_includeloggedon.db')
#show_databases('/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/swap_hsc_online_excludeloggedon.db')

def Information_gained(path):
    def delta_I(M_CL,M_CN,p,C):
      M_CL = M_CL[C]
      M_CN = M_CN[C]
      p_c = p*M_CL + (1-p)*M_CN
      if (p*M_CL/p_c)==0:
        return (1-p)*(M_CN/p_c)*np.log2(M_CN/p_c)
      elif (1-p)*(M_CN/p_c)==0:
        return (p*M_CL/p_c)*np.log2(M_CL/p_c)
      else:
        return (p*M_CL/p_c)*np.log2(M_CL/p_c)+ (1-p)*(M_CN/p_c)*np.log2(M_CN/p_c)

    def S(x):
      if x==0:
        return 0
      else:
        return x*np.log2(x)

    def av_dI(M_CL,M_CN,p):
      M_LL = M_CL[1]
      M_NN = M_CN[0]
      return p*(S(M_LL)+S(1-M_LL))\
        + (1-p)*(S(M_NN)+S(1-M_NN))\
        -S(p*(M_LL)+(1-p)*(1-M_NN))\
        -S(p*(1-M_LL)+(1-p)*M_NN)
    
    user_table=pd.DataFrame.from_dict(read_sqlite(path)['users'])
    print(user_table.columns)
    print(eval(user_table['confusion_matrix'][0])['n_seen'])
    Info =0;Info_list=[]
    for i in tqdm(range(len(user_table))):
#M_CL is p(C) given subject is a lens
      try:
        M_LL = eval(user_table['confusion_matrix'][i])['n_seen'][1]/eval(user_table['confusion_matrix'][i])['n_gold'][1]
      except ZeroDivisionError:
#        print(str(i)+ ', No lens golds seen')
        M_LL = 0.5
      M_NL = 1-M_LL
      try:
        M_NN = eval(user_table['confusion_matrix'][i])['n_seen'][0]/eval(user_table['confusion_matrix'][i])['n_gold'][0]
      except ZeroDivisionError:
#        print(str(i) + ', no dud golds seen')
        M_NN = 0.5
      M_LN = 1-M_NN
#Use p = 0.5 here as want 'normalised' skill (see paper)
      Info_list.append(av_dI([M_NL,M_LL],[M_NN,M_LN],0.5))
      Info+=av_dI([M_NL,M_LL],[M_NN,M_LN],0.5)
    print(Info)
    Info_list = np.array(Info_list)
    print(np.mean(Info_list))

#Information_gained('./data/swap_bFINAL_simul_AWS.db')
#Information_gained('./data/swap_hsc_csv_online.db')

def compare_shuffled_swaps(db_path_0,db_path_shuffled_swaps):
    subject_scores_0 = np.log10(np.array(pd.DataFrame.from_dict(read_sqlite(db_path_0)['subjects'])['score']).astype('float32'))
    subject_subj_id_0= list(pd.DataFrame.from_dict(read_sqlite(db_path_0)['subjects'])['subject_id'])
    fig, ax = pl.subplots(1,len(db_path_shuffled_swaps),figsize=(8*len(db_path_shuffled_swaps),7))
    for f in range(len(db_path_shuffled_swaps)):
      print(f)
      subject_scores_i = np.log10(np.array(pd.DataFrame.from_dict(read_sqlite(db_path_shuffled_swaps[f])['subjects'])['score']).astype('float32'))
    for i in range(len(db_path_shuffled_swaps)):
        subject_scores_i = np.log10(np.array(pd.DataFrame.from_dict(read_sqlite(db_path_shuffled_swaps[i])['subjects'])['score']).astype('float32'))
        subject_subj_id_i = list(pd.DataFrame.from_dict(read_sqlite(db_path_shuffled_swaps[i])['subjects'])['subject_id'])
        score_0_list = np.zeros(len(subject_scores_0));score_1_list = np.zeros(len(subject_scores_0))
#Subject id's may not be in the same order in each database table so need to get correct score for corresponding subject id:
        for k in tqdm(range(len(subject_scores_0))):
          score_0_list[k]= subject_scores_0[k]
          score_1_list[k]= subject_scores_i[subject_subj_id_i.index(subject_subj_id_0[k])]
        h = ax[i].hist2d(score_0_list,score_1_list,bins=np.linspace(-13,0,27),norm=mpl.colors.LogNorm())
        ax[i].set_xlabel('Unshuffled database: log10(score)')
        ax[i].set_ylabel('Shuffled database: log10(score)')
        ax[i].set_title(db_path_shuffled_swaps[i])
#This plots the colourbar - the h[3] index is just where the colour attribute is stored
        pl.colorbar(h[3],ax=ax[i])
    pl.show()

#compare_shuffled_swaps('./data/swap_shuffled_0.db',\
#                        ['./data/swap_shuffled_N100.db','./data/swap_shuffled_N1000.db','./data/swap_shuffled_N10000.db','./data/swap_shuffled_N100000.db','./data/swap_shuffled_N1000000.db'])


def optimising_training_frequency(path):
  def example_training_freq(x,break_val, frac, current):
    if current ==False:
        if x<=break_val:
          return frac[0]
        else:
          return frac[1]
    if current ==True:
        if x<=16:
          return 1/3
        elif 16<x and x<=16+25:
          return 1/5
        else:
          return 1/10

  def cumul_training_freq(N, break_val, frac, current):
    cumul_freq = 0
    for i in range(1,N+1):
      cumul_freq += example_training_freq(i,break_val,frac,current)
    return cumul_freq/N

  user_score=list(pd.DataFrame.from_dict(read_sqlite(path)['users'])['user_score'])
  user_score=[eval(user_score[i]) for i in tqdm(range(len(user_score)))]
  user_score_0 = [user_score[i]['0'] for i in tqdm(range(len(user_score)))]
  user_score_1 = [user_score[i]['1'] for i in tqdm(range(len(user_score)))]
#Histogram of Number of images classified from hsc:
  user_history=pd.DataFrame.from_dict(read_sqlite('./data/swap_hsc_csv_online.db')['users'])['history']
  N_seen = []
#Subtracting 1 as don't want to include the prior in the history:
  for i in tqdm(range(len(user_history))):
    N_seen.append(len(eval(user_history[i]))-1)
  #Needs to be np.linspace(1,np.max(N_seen)+1,np.max(N_seen)+1) (NB starts from 1) here as, from testing, looks like the final bin would group the final two classification bins together if there were just np.max(N_seen) bins. Eg if N_seen went up to 9 (inclusive), would need bins which were np.linspace(1,9+1,9+1) (= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) as final bin is then 9(inclusive)-10.
  N_seen_hist = np.histogram(N_seen,bins=np.linspace(1,np.max(N_seen)+1,np.max(N_seen)+1))[0]
  N_seen_hist = [np.sum(N_seen_hist[i:len(N_seen_hist)]) for i in range(len(N_seen_hist))]
#How fast the user scores increase as they see more training images:
  y_1 = [0., 0.3968254,  0.59090909, 0.66153846, 0.69918699, 0.74912892,0.78658537, 0.77060932, 0.78181818, 0.81818182, 0.83333333,0.85754986,0.8989547,  0.90095238, 0.87162162, 0.8573975,  0.86868687, 0.88516746,0.91212121, 0.92857143, 0.9476584,  0.960047,0.96846847, 0.95444444,0.9337059,  0.93922127, 0.94912846, 0.94702998, 0.95277778, 0.94865375,0.96474359, 0.96784219, 0.97285068,0.96293436, 0.96846847, 0.97370343,0.97866287, 0.97655678, 0.98494709, 0.98885017, 0.99455782, 1.]
#Making the user score increase 2x as slowly as need to see training images from both duds and training-lenses:
  y_2 = [np.mean([y_1[i],y_1[i+1]]) for i in range(len(y_1)-1)]
  y_3 = (y_1+y_2).copy()
#Sorting skill values so user skill increases monotonically (doesn't drastically change learning rate, but smoothes it out a bit)
  y_3.sort()
  y_1 = y_3.copy()
  x_1 = np.arange(0,len(y_1))
#Fitting the change in the user score to an exponential:
#  def exp_func(x,b):
#     return 1-np.exp(x*b)
#  fit_params = cf(exp_func,x_1,y_1)[0]
#  y_1 = exp_func(x_1,fit_params)

  user_convergence_func = interpolate.interp1d(x_1, y_1,bounds_error=False,fill_value = (0,1))
  M_LL_final = np.median(user_score_1)
  M_NN_final = np.median(user_score_0)
  print(M_LL_final,M_NN_final)
  def S(x):
      x = np.array(x)
      y = x*np.log2(x)
      y[x==0]==0
      return y

#NOTE THIS IS A DIFFERENT av_dI FROM ABOVE AS USES SLIGHTLY DIFFERENT ARGUMENTS (EG M_LL NOT M_CL)
  def av_dI(p,M_LL,M_NN):
      return p*(S(M_LL)+S(1-M_LL))\
        + (1-p)*(S(M_NN)+S(1-M_NN))\
        -S(p*(M_LL)+(1-p)*(1-M_NN))\
        -S(p*(1-M_LL)+(1-p)*M_NN)

#n = number of all images seen
  def info_gained(n,break_val,frac,current):
    M_LL = ((M_LL_final-0.5)*user_convergence_func(n*cumul_training_freq(n,break_val,frac,current)))+0.5
    M_NN = ((M_NN_final-0.5)*user_convergence_func(n*cumul_training_freq(n,break_val,frac,current)))+0.5
    return av_dI(0.5,M_LL,M_NN)
  
  def info_gained_tot(break_val,frac,current=False):
      info_tot = 0.0*np.zeros(len(N_seen))
      for i in tqdm(range(1,len(N_seen))):
#Product of the number of people who see the ith image, the fraction of these people for whom this image is a test image, and the information gained by a classification given these people have seen an expected number of training images and thus have a given skill. Note this is the expected info_gained from a average_user, rather than the average of the info_gained from all users. For the latter, see further below.
        info_tot[i] = (N_seen_hist[i]*(1-example_training_freq(i,break_val,frac,current))*info_gained(i,break_val,frac,current))
      return np.sum(info_tot)
  N_seen_hist=np.array(N_seen_hist)
#  pl.plot(np.linspace(1,len(N_seen_hist),len(N_seen_hist)),(N_seen_hist-np.min(N_seen_hist))/(np.max(N_seen_hist)-np.min(N_seen_hist)))
#  q = [];r = []
#  for v in tqdm(range(len(N_seen_hist))):
  #v is the index in N_seen hist (which starts at the first image). v=0 corresponds to the first image, v=2 the second etc.
  #Using current training rate values:
#    q.append(1-example_training_freq(v+1,np.nan, np.nan, True))
#    r.append(info_gained(v+1,np.nan, np.nan, True))
#  q = np.array(q)
#  r = np.array(r)
###  pl.plot(np.linspace(1,len(r),len(r)),q)
#  pl.plot(np.linspace(1,len(r),len(r)),(r-np.min(r))/(np.max(r)-np.min(r)),c='g')
#  tot_info_gained = q*(r-np.min(r))*(N_seen_hist-np.min(N_seen_hist))/((np.max(r)-np.min(r))*(np.max(N_seen_hist)-np.min(N_seen_hist)))
#  pl.plot(np.linspace(1,len(r),len(r)),tot_info_gained,c='k')
#  pl.legend(['N. users','Info gained/user','Total info gained'])
#  pl.fill_between(x= np.linspace(1,len(r),len(r)),y1= tot_info_gained,color= "k",alpha= 0.3)
#  pl.xlabel('Image number')
#  pl.ylabel('Normalised Scale')
#  pl.show()
  def info_from_av_user():
#      break_val_list = [5,10,15,20,40,60,80,100,120]
      break_val_list = [5,10,40]
      break_val_colors = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
      fig, ax = pl.subplots(3,3,figsize=(10,10))
      frac_list=[[0.1,0.2],[0.2,0.3]]
      frac_list=[]
      for t in range(1,6):
        for p in range(5):
          frac_list.append([0.2*t,0.1*p])
      Z_list = []
      for u in range(len(break_val_list)):
          X=[];Y=[];Z=[]
          for k in range(len(frac_list)):
            frac = frac_list[k]
            X.append(frac[0]);Y.append(frac[1]);Z.append(info_gained_tot(break_val_list[u],frac))
          Z_list.append(Z)
      Z_current = info_gained_tot(break_val_list[u],frac,True)
      Z_list = np.array(Z_list)
      Z_min = np.min([np.min(Z_list),Z_current]); Z_max = np.max([np.max(Z_list),Z_current])
      txt='Current value: ' + str(np.round((Z_current-Z_min)/(Z_max-Z_min),2))
      pl.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
      Z_list = (Z_list-Z_min)/(Z_max-Z_min)
      for u in range(len(break_val_list)):
          if u<3:
            f = 0
            g = u
          elif u<6:
            f = 1
            g = u-3
          else:
            f = 2
            g = u-6
          Z = Z_list[u]
          for r in range(len(frac_list)):
            ax[f,g].annotate(str(np.round(Z[r],2)), (X[r],Y[r]))
          Z = 0.2+0.8*Z
          ax[f,g].scatter(X,Y,c=Z,cmap=break_val_colors[u])
          ax[f,g].axis('scaled')
          ax[f,g].set_title('Break Value: ' + str(break_val_list[u]))
          ax[f,g].set_xlabel('Initial Training Fraction')
          ax[f,g].set_ylabel('Final Training Fraction')
      pl.show()

#  info_from_av_user()

  def user_training_sample(N,break_val,frac,current):
     if current==False:
        if N<=break_val:
          training_indx =  np.random.choice([0,1],size=N,p = [1-frac[0],frac[0]])
          return training_indx,np.cumsum(training_indx)
        else:
          training_indx =  np.concatenate((np.random.choice([0,1],size=break_val,p = [1-frac[0],frac[0]]),np.random.choice([0,1],size=N-break_val,p = [1-frac[1],frac[1]])))
          return training_indx,np.cumsum(training_indx)
     if current==True:
        if N<=16:
          training_indx =  np.random.choice([0,1],size=N,p = [2/3,1/3])
          return training_indx,np.cumsum(training_indx)
        elif N<=25+16:
          training_indx =  np.concatenate((np.random.choice([0,1],size=16,p = [2/3,1/3]),np.random.choice([0,1],size=N-16,p = [4/5,1/5])))
          return training_indx,np.cumsum(training_indx)
        else:
          training_indx =  np.concatenate((np.random.choice([0,1],size=16,p = [2/3,1/3]),np.random.choice([0,1],size=25,p = [4/5,1/5]),np.random.choice([0,1],size=N-16-25,p = [9/10,1/10])))
          return training_indx,np.cumsum(training_indx)

  N_seen_hist_norm = np.histogram(N_seen,bins=np.linspace(0,np.max(N_seen)+1,np.max(N_seen)+2),density=True)[0]
  
  #Samples random users, calculates the total information they provide based on the number of images they saw, how fast the typical user score increases with training images, and the number of test images seen. Returns I_total = Sum[Info_gained_from_image*BOOL(Test_image?)]
  def random_user_sampling(iteration,frac,break_val,current=False):
     tot_info = 0
     N_array = np.random.choice(np.arange(len(N_seen_hist_norm)), size=iteration, replace=True, p=N_seen_hist_norm)
     if current==False:
         for i in range(iteration):
            N = N_array[i]
            training_indx,cumul_training_indx = user_training_sample(N,break_val,frac,current)
    #        print('Cumulative training index')
    #        print(cumul_training_indx)
    #        print('user convergence func')
    #        print(user_convergence_func(cumul_training_indx))
    #        print('User skill')
            M_LL = ((M_LL_final-0.5)*user_convergence_func(cumul_training_indx))+0.5
            M_NN = ((M_NN_final-0.5)*user_convergence_func(cumul_training_indx))+0.5
    #        print(M_LL)
    #        print(M_NN)
    #        print('average info')
#Returns I_total = Sum[Info_gained_from_image*BOOL(Test_image?)]
            av_info =np.sum(av_dI(0.5,M_LL,M_NN)*(1-training_indx))
            tot_info+=av_info
     else:
        for i in tqdm(range(iteration)):
            N = N_array[i]
            training_indx,cumul_training_indx =user_training_sample(N,break_val,frac,current)
            M_LL = ((M_LL_final-0.5)*user_convergence_func(cumul_training_indx))+0.5
            M_NN = ((M_NN_final-0.5)*user_convergence_func(cumul_training_indx))+0.5
            av_info =np.sum(av_dI(0.5,M_LL,M_NN)*(1-training_indx))
            tot_info+=av_info
     return tot_info

  
  def plot_info():
      break_val_list = [5,10,15,20,40,60,80,100,120]
      break_val_colors = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
      fig, ax = pl.subplots(3,3,figsize=(10,10))
      frac_list=[[0.1,0.2],[0.2,0.3]]
      frac_list=[]
      for t in range(1,6):
        for p in range(5):
          frac_list.append([0.2*t,0.1*p])
      Z_list = []
      N_runs = 100000
      for u in tqdm(range(len(break_val_list))):
          X=[];Y=[];Z=[]
          for k in range(len(frac_list)):
            frac = frac_list[k]
            X.append(frac[0]);Y.append(frac[1]);Z.append(random_user_sampling(N_runs,frac,break_val_list[u]))
#            X.append(frac[0]);Y.append(frac[1]);Z.append(random_user_sampling(100000,frac,break_val_list[u]))
          Z_list.append(Z)
          print(X)
          print(Y)
          print(Z)
#          np.save('/Users/hollowayp/Documents/Coding_Files/Files for 1st Year Presentation/b_'+str(break_val_list[u]),Z)
#      Z_current = np.min(Z_list) #Remove this line once calculated actual Z_current
#      Z_current = info_gained_tot(np.nan,frac,True)
      Z_current = random_user_sampling(N_runs,np.nan,np.nan,current=True)
      Z_list = np.array(Z_list)
      Z_min = np.min([np.min(Z_list),Z_current]); Z_max = np.max([np.max(Z_list),Z_current])
      txt='Current value: ' + str(np.round((Z_current-Z_min)/(Z_max-Z_min),2))
      pl.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
      Z_list = (Z_list-Z_min)/(Z_max-Z_min)
      for u in range(len(break_val_list)):
          if u<3:
            f = 0; g = u
          elif u<6:
            f = 1;g = u-3
          else:
            f = 2;g = u-6
          Z = Z_list[u]
          for r in range(len(frac_list)):
            ax[f,g].annotate(str(np.round(Z[r],2)), (X[r],Y[r]))
          Z = 0.2+0.8*Z
          ax[f,g].scatter(X,Y,c=Z,cmap=break_val_colors[u])
          ax[f,g].axis('scaled')
          ax[f,g].set_title('Break Value: ' + str(break_val_list[u]))
          ax[f,g].set_xlabel('Initial Training Fraction')
          ax[f,g].set_ylabel('Final Training Fraction')
      print(Z_min,Z_max,Z_current)
#      np.save('/Users/hollowayp/Documents/Coding_Files/Files for 1st Year Presentation/min_max_current_val',np.array([Z_min,Z_max,Z_current]))
      pl.show()
    
#  def calculate_current_info():
#    #Note number of iterations will change info value, so need to make sure it is the same as above:
#    print(random_user_sampling(100000,[0,0],0,current=True))
#    print(random_user_sampling(100,[0,0],0,current=True))

  plot_info()
#  calculate_current_info()

#optimising_training_frequency('/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/swap_bFINAL_hardsimsaretest.db')
#optimising_training_frequency('/Users/hollowayp/Documents/GitHub/kSWAP/examples/data/swap_bFINAL_hardsimsaretest_excludenotloggedon_changingalreadyseen3tofalse.db')

import json
def hsc_high_score_users(path):
  users =pd.DataFrame.from_dict(read_sqlite(path)['users'])
  user_id = users['user_id']
  user_score_matrices=users['user_score']
  user_confusion_matrices=users['confusion_matrix']
  # start with a square Figure
  fig = pl.figure(figsize=(8, 8))
  xy = [];s=[]
  for p in tqdm(range(len(user_score_matrices))):
    xy.append(eval(user_score_matrices[p]))
    s.append(eval(user_confusion_matrices[p]))
#          x_NL.append(eval(user_score_matrices[p])['1']);y_NL.append(eval(user_score_matrices[p])['0']);s_NL.append(sum(eval(user_confusion_matrices[p])['n_seen']))
  xy = np.array(xy)
  s = np.array(s)
  x = np.array([xy[i]['1'] for i in tqdm(range(len(xy)))])
  y = np.array([xy[i]['0'] for i in tqdm(range(len(xy)))])
  s = np.array([np.sum(s[i]['n_gold']) for i in tqdm(range(len(s)))])
  classification_limit = 500
  s_large = s[s>classification_limit]
  pl.scatter(x[s>classification_limit], y[s>classification_limit],s=(1+s_large)/15,c='blue')
  pl.gca().set_aspect('equal')
  pl.xlabel('P("LENS"|LENS)')
  pl.ylabel('P("NOT"|NOT)')
  pl.xlim(0,1)
  pl.ylim(0,1)
  print(user_id[(s>classification_limit) & (x>0.8)& (y>0.8)])
  user_id_list = [82,1775810,1533,117105]
  for i in range(len(users)):
    if user_id[i] in user_id_list:
        print(user_score_matrices[i],user_confusion_matrices[i])
  pl.show()

#hsc_high_score_users('./data/swap_hsc_online_includeloggedon.db')
#hsc_high_score_users('./data/swap_hsc_online_excludeloggedon.db')
#hsc_high_score_users('./data/swap_hsc_csv_online.db')


def N_class_func(path):
    print(path)
    users =pd.DataFrame.from_dict(read_sqlite(path)['users'])
#    print(pd.DataFrame.from_dict(read_sqlite(path)['subjects']))
    subjects_hist = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])
    subjects_gold =  pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['gold_label']
    N_class_outside_retirement_bounds = 0
    N_class_all = 0
    for i in range(len(subjects_hist)):
      if subjects_gold[i]==-1:
        subj_hist_i = eval(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['history'][i])
        for j in range(len(subj_hist_i)):
          N_class_all+=1
          if subj_hist_i[j][3]>10**-5:
            N_class_outside_retirement_bounds+=1
#    for i in range(len(users)):
#      N_class+=len(eval(users['history'][i]))-1
    print(N_class_all,N_class_outside_retirement_bounds)

path_list = ['./data/swap_bFINAL_hardsimsaretest.db','./data/swap_bFINAL_hardsimsaretest_excludenotloggedon.db','./data/swap_bFINAL_hardsimsaretest_excludenotloggedon_changingalreadyseen3tofalse.db','./data/swap_bFINAL_hardsimsaretest_excludenotloggedon_changingalreadyseen3tofalse_offlineswap.db','./data/swap_bFINAL_hardsimsaretraining.db','./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon.db','./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon_changingalreadyseen3tofalse.db','./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon_changingalreadyseen3tofalse_offlineswap.db','./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon_offlineswap.db','./data/swap_bFINAL_hardsimsaretraining_offlineswap.db','./data/swap_bFINAL_simul_AWS.db','/Users/hollowayp/Documents/swap beta25 backup1.db','/Users/hollowayp/Documents/swap beta25 backup2.db','/Users/hollowayp/Documents/swap beta25 backup3.db','/Users/hollowayp/Documents/swap beta25 backup4.db','/Users/hollowayp/Documents/swap beta25 backup5.db','/Users/hollowayp/Documents/swap beta25 backup6.db','/Users/hollowayp/Documents/swap beta25 backup7.db','/Users/hollowayp/Documents/swap beta25 backup8.db','./data/swap_simul_AWS.db','./data/swap_simul_hardsimsaretest_excludenotloggedon.db','./data/swap_simul_hardsimsaretest.db','./data/swap_simul_hardsimsaretraining_excludenotloggedon_changingalreadyseen3tofalse_offlineswap.db','./data/swap_simul_hardsimsaretraining_excludenotloggedon_changingalreadyseen3tofalse.db','./data/swap_simul_hardsimsaretraining_excludenotloggedon_offlineswap.db','./data/swap_simul_hardsimsaretraining_excludenotloggedon.db','./data/swap_simul_hardsimsaretraining.db']

for i in range(0):
  path_i = path_list[i]
  N_class_func(path_i)

def plot_histogram_with_text(x,y,z,shape,xlabel = '',ylabel = '',title = '',figtxt = '',min_z = np.nan,max_z = np.nan):
    if np.isnan(min_z):
      min_z = np.min(z)
    if np.isnan(max_z):
      max_z = np.max(z)
    dx = 0.5*(np.min(x[x!=np.min(x)])-np.min(x))
    dy = 0.5*(np.min(y[y!=np.min(y)])-np.min(y))
    x = x.reshape(shape,order = 'F')
    y = y.reshape(shape,order = 'F')
    z = z.reshape(shape,order = 'F')
    z = np.round(z,2)
    # The normal figure
    fig = pl.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    im = ax.imshow(z, extent=[np.min(x)-dx,np.max(x)+ dx,np.min(y)-dy,np.max(y)+dy], origin='lower',aspect='auto',norm = colors.Normalize(min_z,max_z))
    # Add the text
    jump_x = (np.max(x) - np.min(x)) / (2.0 * (len(x)-1))
    jump_y = (np.max(y) - np.min(y)) / (2.0 * (len(y)-1))
    x_positions = np.linspace(start=np.min(x)-dx, stop=np.max(x)+dx, num=len(x), endpoint=False)
    y_positions = np.linspace(start=np.min(y)-dy, stop=np.max(y)+dy, num=len(y), endpoint=False)
    for y_index, y in enumerate(y_positions):
        for x_index, x in enumerate(x_positions):
            label = z[y_index, x_index]
            text_x = x + jump_x
            text_y = y + jump_y
            if y_index<2: #y index starts counting from the bottom
                ax.text(text_x, text_y, label, color='black', ha='center', va='center',fontsize=13)
            else:
                ax.text(text_x, text_y, label, color='white', ha='center', va='center',fontsize=13)
#    fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(min_z,max_z)))
    pl.xlabel(xlabel,fontsize=15)
    pl.ylabel(ylabel,fontsize=15)
    pl.title(title)
    pl.figtext(0.5, 0.01, figtxt, wrap=True, horizontalalignment='center', fontsize=12)
    pl.savefig('/Users/hollowayp/Documents/Coding_Files/Files for 1st Year Presentation/'+str(figtxt),transparent=True,dpi = 1000)
#    pl.show()

a = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.6000000000000001, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0])
b = np.array([0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.0, 0.1, 0.2, 0.30000000000000004, 0.4, 0.0, 0.1, 0.2, 0.30000000000000004, 0.4])
min_z,max_z,cur_z = np.load('/Users/hollowayp/Documents/Coding_Files/Files for 1st Year Presentation/min_max_current_val.npy')
break_vals = [5,10,15,20,40,60,80,100,120]
for j in range(len(break_vals)):
  i = break_vals[j]
  c = np.load('/Users/hollowayp/Documents/Coding_Files/Files for 1st Year Presentation/b_'+str(i)+'.npy')
  c = (c-min_z)/(max_z-min_z)
  plot_histogram_with_text(a,b,c,(5,5),'Initial Training Proportion','Final Training Proportion', '', 'Break Value: ' + str(i),0,1)
