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
#First if: Retirement could only be inefficient if some of the scores are beyond a threshold, and the subject isn't a gold...
#Second if: and only if there are any occurences when the score is beyond the threshold (LHS, below) which is not the last time a classification is made (RHS)
#Third if: and only if two classifications have been made after the subject has been seen more than lower_retirement_limit times:
      if (history_scores<p_retire_lower).any() and subject_golds[i]==-1:
        if np.where(history_scores<p_retire_lower)[0][0]!=len(history_scores)-1:
          if np.sum((history_scores<p_retire_lower)*[1 if i >= lower_retirement_limit else 0 for i in range(len(history_scores))])>=2:
            inefficiency_count_i=1
#Repeating for the other threshold:
      if (history_scores>p_retire_upper).any() and subject_golds[i]==-1:
        if np.where(history_scores>p_retire_upper)[0][0]!=len(history_scores)-1:
          if np.sum((history_scores>p_retire_upper)*[1 if i >= lower_retirement_limit else 0 for i in range(len(history_scores))])>=2:
            inefficiency_count_i=1
      inefficiency_count+=inefficiency_count_i
      if inefficiency_count_i==1:
        should_be_retired_list.append(i)
    print(inefficiency_count,should_be_retired_list)
    return should_be_retired_list
should_be_retired_list= efficiency_calc()

#FILE PATH HERE
def trajectory_plot(path='./data/swap.db', subjects=[]):
    print(pd.DataFrame.from_dict(read_sqlite(path)['subjects']))
    subject_id=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id']
    subject_histories = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['history']
    histories_df = pd.DataFrame(subject_histories)
    subject_golds=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['gold_label']
    retired=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['retired']
    retired = sum(1 for x in retired if x!=0)
    users =pd.DataFrame.from_dict(read_sqlite(path)['users'])
    user_score_matrices=users['user_score']
    user_confusion_matrices=users['confusion_matrix']

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
    if type(subjects) == int:
# draw random numbers (note there can be duplication) or plot all of them
#        while len(subjects_final) < subjects:
#            indx = np.random.choice(len(subject_id))
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
            classification_time_i[0]=-100
            classification_time_final.append(classification_time_i)
            if len(history_i) > max_seen:
                max_seen = len(history_i)-1
    else:
      for p in range(len(subjects)):
        indx = p
        subjects_final.append(subjects[indx])
        golds_final.append(subject_golds[indx])
        history_i=[]
        classification_time_i=[]
        history_all_i=[]
        history_list_i=eval(subject_histories[indx])
        for i in range(len(history_list_i)):
          history_i.append(history_list_i[i][3])
          history_all_i.append(history_list_i[i])
          date_i=history_list_i[i][5]
          if i==0:
            classification_time_i.append(-100)
          else:
            classification_time_i.append(datetime.datetime(int(date_i[0:4]),int(date_i[5:7]),int(date_i[8:10]),int(date_i[11:13]),int(date_i[14:16]),int(date_i[17:19])).timestamp())
        subjects_history_final.append(history_i)
        history_all_final.append(history_all_i)
        classification_time_final.append(np.array(classification_time_i)-classification_time_i[1])
        if len(history_i) > max_seen:
            max_seen = len(history_i)
    subjects = subjects_final
    p_real , p_bogus= thresholds_setting()

    def plotting_traj(subjects, subjects_history_final,timer=False):
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
            ax.set_xlabel('Posterior Probability Pr(LENS|d)',fontsize=5)
            ax.set_ylabel('Number of Classifications',fontsize=5)
            # plot history trajectories
            sub_list=[]
            for j in range(len(subjects)):
                # clip history
                history = np.array(subjects_history_final[j])
#                history = np.where(history < p_min, p_min, history)
#                history = np.where(history > p_max, p_max, history)
                # trajectory
                y = np.arange(len(history) + 1)-1
                # add initial value
                history = np.append(prior_setting(), history)
                ax.plot(history, y, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],linewidth=0.5)
                ax.scatter(history, y,marker='+',linewidth=1,color=colors[golds_final[j]],alpha=0.6,s=0.5)

                # add a point at the end
                ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,edgecolors=colors[golds_final[j]],facecolors=colors[golds_final[j]])
                if j in should_be_retired_list:
                  ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,c='k')
                  abc=[]
                  for s in range(len(classification_time_final[j])):
                    abc.append(str(datetime.timedelta(seconds=classification_time_final[j][s])))
                  print(abc)
            patches = []
            for color, alpha, label in zip(colors, alphas, ['Bogus', 'Real', 'Test']):
                patch = mpatches.Patch(color=color, alpha=alpha, label=label)
                patches.append(patch)
            ax.legend(handles=patches, loc='upper right', framealpha=1.0,prop={'size':5})
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
#                history = np.where(history < p_min, p_min, history)
#                history = np.where(history > p_max, p_max, history)
                y = classification_time_final[j]
                ax.plot(history[1:len(history)], y[1:len(y)]+1, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],linewidth=0.5)
#                for f in range(1,len(y)):
#                  ax.annotate(datetime.timedelta(seconds=y[f]), (history[f], y[f]),fontsize=3)
                ax.scatter(history, y,marker='+',linewidth=1,color=colors[golds_final[j]],alpha=1,s=1)
                ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,edgecolors=colors[golds_final[j]],facecolors=colors[golds_final[j]])
                if j in should_be_retired_list:
                  ax.scatter(history, y,marker='+',linewidth=1,color='k',s=1)
                  ax.scatter(history[-1:], y[-1:], s=1,color='k')
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
        pl.hist(N_class_bins[r],bins=(max(N_class_final)-min(N_class_final))+1,range=(min(N_class_final),max(N_class_final)+1),                      stacked=False,edgecolor=edgecolors[r],fill=False,align='left')
      pl.xlabel('Number of Classifications')
      pl.ylabel('N')
      pl.legend(['Test','Dud','Lens'])
      pl.show()
      post_prob_bins=[[],[],[]]
      for j in range(len(posterior_prob_final)):
        if len(subjects_history_final[j])!=1:
          post_prob_bins[golds_final[j]+1].append(posterior_prob_final[j])
      pl.hist(post_prob_bins[0],bins=np.logspace(np.log10(1e-8),np.log10(1.0), 20),stacked=False,edgecolor = 'grey',fill=False)
      pl.hist(post_prob_bins[1],bins=np.logspace(np.log10(1e-8),np.log10(1.0), 20),stacked=False,edgecolor='red',fill=False)
      pl.hist(post_prob_bins[2],bins=np.logspace(np.log10(1e-8),np.log10(1.0), 20),stacked=False,edgecolor='blue',fill=False)
      pl.legend(['Test','Dud','Lens'])
      pl.xlabel('Posterior Probability')
      pl.ylabel('N')
      pl.yscale('log')
      pl.xscale('log')
      pl.show()
      for p in range(len(user_score_matrices)):
        pl.scatter(eval(user_score_matrices[p])['1'],eval(user_score_matrices[p])['0'],s=(1+sum(eval(user_confusion_matrices[p])['n_seen']))/15,color='blue')
      pl.gca().set_aspect('equal')
      pl.xlabel('P("LENS"|LENS)')
      pl.ylabel('P("NOT|NOT)')
      pl.xlim(0,1)
      pl.ylim(0,1)
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

trajectory_plot()
