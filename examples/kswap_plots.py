import sys
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.pyplot as pl
import numpy as np
import datetime
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
    import sqlite3
    from pandas import read_sql_query, read_sql_table
    with sqlite3.connect(dbfile) as dbcon:
      tables = list(read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
      out = {tbl : read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}
    return out

def date_time_convert(self, row_created_at):
  s_i=[]
  for s in range(len(row_created_at)):
    if row_created_at[s]=='-' or row_created_at[s]==':':
      s_i.append(s)
  dt_i = datetime(int(row_created_at[0:s_i[0]]), int(row_created_at[s_i[0]+1:s_i[1]]), int(row_created_at[s_i[1]+1:s_i[1]+3]),int(row_created_at[s_i[2]-2:s_i[2]]), int(row_created_at[s_i[2]+1:s_i[3]])).timestamp()
  return dt_i

def efficiency_calc(path='./data/swap.db'):
    from config import Config as config_0
    p_retire_lower=config_0().thresholds[0]
    p_retire_upper=config_0().thresholds[1]
    lower_retirement_limit=config_0().lower_retirement_limit
    retirement_limit=config_0().retirement_limit
#    print(pd.DataFrame.from_dict(read_sqlite(path)['subjects']))
    user_table=pd.DataFrame.from_dict(read_sqlite(path)['users'])['history']
    subject_histories = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['history']
    subject_golds=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['gold_label']
    inefficiency_count=0
    should_be_retired_list=[]
    for i in range(len(subject_histories)):
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
    final_classification_user={}
    fraction_of_total_user_classifications={}
    delta_time_dict = {}
    for p in should_be_retired_list:
        subj_hist=eval(subject_histories[p])
        subj_id =pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id'][p]
        final_classification_time[subj_id]= subj_hist[len(subj_hist)-1][5]
        penultimate_classification_time[subj_id] = subj_hist[len(subj_hist)-2][5]
    for s in range(len(user_table)):
        subjects_seen_by_user=[]
        user_history=eval(pd.DataFrame.from_dict(read_sqlite(path)['users'])['history'][s])
        for q in range(len(user_history)):
            subjects_seen_by_user.append(user_history[q][0])
        for r in range(len(subject_ids_which_should_be_retired)):
            if subject_ids_which_should_be_retired[r] in subjects_seen_by_user:
              indx_of_classification=subjects_seen_by_user.index(subject_ids_which_should_be_retired[r])
              if user_history[indx_of_classification][3]==final_classification_time[subject_ids_which_should_be_retired[r]]:
                assert subject_ids_which_should_be_retired[r] not in final_classification_user
                final_classification_user[subject_ids_which_should_be_retired[r]]=pd.DataFrame.from_dict(read_sqlite(path)['users'])['user_id'][s]
                user_dict_key = 'user_indx: '+str(s)+' total: ' + str(len(subjects_seen_by_user))
                user_dict_key_2 = str(indx_of_classification)+ '/'+str(len(subjects_seen_by_user))
                if user_dict_key in fraction_of_total_user_classifications:
                  fraction_of_total_user_classifications[user_dict_key].append(indx_of_classification)
                  delta_time_dict[user_dict_key_2].append(final_classification_time[subject_ids_which_should_be_retired[r]]-penultimate_classification_time[subject_ids_which_should_be_retired[r]])
                else:
                  fraction_of_total_user_classifications[user_dict_key] = [indx_of_classification]
                  delta_time_dict[user_dict_key_2]= final_classification_time[subject_ids_which_should_be_retired[r]]-penultimate_classification_time[subject_ids_which_should_be_retired[r]]
    print('Wasted classifications: ' + str(inefficiency_count))
    print(fraction_of_total_user_classifications)
#    print(check)
    return should_be_retired_list
#(efficiency_calc('./data/swap_bFINAL_hardsimsaretest_excludenotloggedon.db'))
#(efficiency_calc('./data/swap_bFINAL_hardsimsaretraining_excludenotloggedon.db'))
#(efficiency_calc('./data/swap_bFINAL_simul_AWS.db'))

from tqdm import tqdm
import matplotlib as mpl

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

def gold_frequency(path):
    user_table=pd.DataFrame.from_dict(read_sqlite(path)['users'])['history']
    print(eval(user_table[0])[0:5])
#    user_score=list(pd.DataFrame.from_dict(read_sqlite(path)['users'])['user_score'])
#    user_score=[eval(user_score[i]) for i in range(len(user_score))]
#    user_score=[(user_score[i]["0"]**2+user_score[i]["1"]**2)**0.5 for i in range(len(user_score))]
    subject_histories = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['history']
    subject_ids=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id']
    subject_golds=np.array(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['gold_label'])
    gold_ids= list(subject_ids[subject_golds!=-1])
#    time_first_classification=[]
#    for s in range(len(user_table)):
#      time_first_classification.append(eval(user_table[s])[1][2])
    min_time=eval(user_table[0])[1][2]
    max_time=0
    for x in tqdm(range(len(user_table))):
      for y in range(1,len(eval(user_table[x]))):
        min_time= np.min([min_time,eval(user_table[x])[y][2]])
        max_time=np.max([max_time,eval(user_table[x])[y][2]])
#    user_score=(user_score-np.min(user_score))/(np.max(user_score)-np.min(user_score))
    cmap = pl.cm.cool
    pl.plot(np.arange(1,800),expected_training_fraction_func(np.arange(1,800)),c='k')
    from matplotlib import cm
    N_less = np.zeros(50);N_more = np.zeros(50)
    for i in tqdm(range(0,len(user_table))):
      x_i = [];y_i = [];t_i = []
#Has to start at 1 as the zeroth index is the prior:
      for j in range(1,len(eval(user_table[i]))):
        if j ==1:
            if eval(user_table[i])[j][0] in gold_ids:
              y_i.append(1)
              x_i.append(j)
              t_i.append(eval(user_table[i])[j][2])
            else:
              y_i.append(0)
              x_i.append(j)
              t_i.append(eval(user_table[i])[j][2])
        else:
            if eval(user_table[i])[j][0] in gold_ids:
              y_i.append(y_i[len(y_i)-1]+1)
              x_i.append(j)
              t_i.append(eval(user_table[i])[j][2])
            else:
              y_i.append(y_i[len(y_i)-1])
              x_i.append(j)
              t_i.append(eval(user_table[i])[j][2])
      x_i = np.array(x_i)
      y_i = np.array(y_i)/x_i
      t_i = np.array(t_i)
      t_i = (t_i-min_time)/(max_time-min_time)
      pl.scatter(x_i,y_i, c=cm.cool(t_i), edgecolor='none',s=15,alpha=1)
      pl.plot(x_i,y_i, color='gray',alpha=0.3)
#      pl.plot(x_i,y_i,c=cmap.to_rgba(t_i))
      if len(x_i)>50:
        N_more += (y_i[0:50]>expected_training_fraction_func(np.arange(1,51))).astype('int')
        N_less += (y_i[0:50]<expected_training_fraction_func(np.arange(1,51))).astype('int')
    norm = pl.Normalize(0, 1)
    sm = pl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = pl.colorbar(sm, ax=None, orientation='vertical')
    pl.xlabel('Number of classifications made by user')
    pl.ylabel('Running fraction of classifications made which are golds')
    pl.show()
    pl.plot(N_more)
    pl.plot(N_less)
    pl.legend(['More','Less'])
    pl.show()

#gold_frequency('./data/swap_bFINAL_simul_AWS.db')
#Just looking at first ~12 hours of beta test to see how many golds were shown in early stages. This is also better for colour of scatter points as they are less bunched up near early times (for a long run, most classifications were in the first day, but plotting with uniform time up to say ~7 days).
#gold_frequency('/Users/hollowayp/Documents/swap beta25 backup3.db')

def score_plots(path='./data/swap_bFINAL_hardsimsaretest.db'):
    user_history=pd.DataFrame.from_dict(read_sqlite(path)['users'])['history']
    f_i = 0;f_max = 0
    for f in tqdm(range(len(user_history))):
      if len(eval(user_history[f]))>f_max:
        f_max = len(eval(user_history[f]))
        f_i = f
    user_score_lens = [[eval(user_history[i])[j][1]["1"] for j in range(len(eval(user_history[i])))] for i in tqdm(range(len(user_history)))]
    user_score_dud = [[eval(user_history[i])[j][1]["0"] for j in range(len(eval(user_history[i])))] for i in tqdm(range(len(user_history)))]
    delta_user_score_lens = [[(user_score_lens[i][j+1]-user_score_lens[i][j]) for j in range(len(user_score_lens[i])-1)] for i in tqdm(range(len(user_history)))]
    delta_user_score_dud = [[(user_score_dud[i][j+1]-user_score_dud[i][j]) for j in range(len(user_score_dud[i])-1)] for i in tqdm(range(len(user_history)))]
    fig, ax = pl.subplots(2,3,figsize=(25,10))
    N=[]
    frac_lens_array = []
    for k in tqdm(range(len(delta_user_score_lens))):
      delta_user_score_lens[k] = np.array(delta_user_score_lens[k])
      delta_user_score_lens[k] = delta_user_score_lens[k][delta_user_score_lens[k]!=0]
      delta_user_score_lens[k] = [0]+delta_user_score_lens[k]
      N.append(len(delta_user_score_lens[k]))
      ax[0,0].plot(user_score_lens[k])
      ax[0,1].plot(abs(delta_user_score_lens[k]),alpha=0.3)
      if len(delta_user_score_lens[k])>=20:
        delta_tot = np.sum(delta_user_score_lens[k])
        frac = [np.sum(delta_user_score_lens[k][0:i]) for i in range(1,len(delta_user_score_lens[k])+1)]
#Adding a '0' so the x axis corresponds to number of training subjects seen.
        frac = [0]+frac
        frac_lens_array.append(np.array(frac)/delta_tot)
        ax[0,2].plot(np.array(frac)/delta_tot,alpha=0.4)
    print(sum(np.array(N)>=20))
    N=[]
    frac_dud_array = []
    for k in tqdm(range(len(delta_user_score_dud))):
      delta_user_score_dud[k] = np.array(delta_user_score_dud[k])
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
    ax[0,0].plot(user_score_lens[f_i],color='k')
    ax[0,1].plot(abs(delta_user_score_lens[f_i]),color='k',alpha=0.7)
    ax[1,0].plot(user_score_dud[f_i],color='k')
    ax[1,1].plot(abs(delta_user_score_dud[f_i]),color='k',alpha=0.7)
    for p in range(2):
        ax[p,0].set_ylabel('User score')
        ax[p,0].set_xlabel('Number of subjects seen')
        ax[p,1].set_ylabel('Change in user score')
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
    for i in range(len(subject_id)):
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
    for f in range(len(subject_id)):
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
    for indx in range(len(subject_id)):
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
        p_min = 5e-8
        p_max = 1
        if timer==False:
            ax.set_xlim(p_min, p_max)
            ax.set_xscale('log')
            ax.set_ylim(max_seen+1,0)
            ax.axvline(x=prior_setting(), color=color_test, linestyle='dotted')
            ax.axvline(x=p_bogus, color=color_bogus, linestyle='dotted')
            ax.axvline(x=p_real, color=color_real, linestyle='dotted')
            ax.fill_betweenx(x1=p_min,x2=p_bogus,y=[max_seen+1,config_0().lower_retirement_limit],alpha=0.2,color='darkviolet')
            ax.fill_betweenx(x1=p_bogus,x2=p_max,y=[max_seen+1,config_0().retirement_limit],alpha=0.2,color='darkviolet')
            ax.set_xlabel('Posterior Probability Pr(LENS|d)',fontsize=5)
            ax.set_ylabel('Number of Classifications',fontsize=5)
            sub_list=[]
            print(str(len(should_be_retired_list)) + ' subjects should have been retired but were subsequently classified')
            for j in range(len(subjects)):
                history = np.array(subjects_history_final[j])
                y = np.arange(len(history) + 1)-1
                history = np.append(prior_setting(), history)
                if history[len(history)-1]<10**-2 and history[len(history)-1]>10**-3 and golds_final[j]==1:
                  print(subjects[j],j)
                if j not in should_be_retired_list:
                    ax.plot(history, y, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],linewidth=0.5)
                    ax.scatter(history, y,marker='+',linewidth=1,color=colors[golds_final[j]],alpha=0.6,s=0.5)
                    # add a point at the end
                    ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,edgecolors=colors[golds_final[j]],facecolors=colors[golds_final[j]])
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
            for j in range(len(subjects)):
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
      pl.legend(['Test','Dud','Lens'])
      pl.show()
      post_prob_bins=[[],[],[]]
      for j in range(len(posterior_prob_final)):
        if len(subjects_history_final[j])!=1:
          post_prob_bins[golds_final[j]+1].append(posterior_prob_final[j])
      pl.hist(post_prob_bins[0],bins=np.logspace(np.log10(1e-8),np.log10(1.0), 20),stacked=False,edgecolor = 'grey',fill=False,density=True)
      pl.hist(post_prob_bins[1],bins=np.logspace(np.log10(1e-8),np.log10(1.0), 20),stacked=False,edgecolor='red',fill=False,density=True)
      pl.hist(post_prob_bins[2],bins=np.logspace(np.log10(1e-8),np.log10(1.0), 20),stacked=False,edgecolor='blue',fill=False,density=True)
      pl.legend(['Test','Training: Dud','Training: Lens'])
      pl.xlabel('Posterior Probability')
      pl.ylabel('N')
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
#AWS swap
#should_be_retired_list= efficiency_calc('./data/swap_bFINAL_simul_AWS.db')
#should_be_retired_list=[]
#trajectory_plot('./data/swap_bFINAL_simul_AWS.db')
#trajectory_plot('/Users/hollowayp/Documents/swap beta25 backup2.db')
#trajectory_plot('./data/swap_hsc_csv_online.db')

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
  import multiprocess as mp
  import time
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
  
plotting_user_score('./data/swap_hsc_csv_online.db')


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
    import csv
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

from collections import Counter
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
    print(list(subjects['seen'][0:110]))
    print(list(subjects[(subjects['gold_label']!=0) & (subjects['score']>10**-5) ]['subject_id']))

#show_databases('./data/swap_hardsimscode_demo.db')
#show_databases('./data/swap_bFINAL_simul_AWS.db')
