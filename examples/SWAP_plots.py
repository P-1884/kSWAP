from pandas import read_sql_query
import matplotlib.patches as mpatches
import matplotlib.pyplot as pl
import pandas as pd
import numpy as np
import sqlite3
import sys
from matplotlib import cm, colors
from tqdm import tqdm

sys.path.append('PATH/TO/CONFIG/FILE/DIRECTORY')

from SWAP_Config_DES import Config

def thresholds_setting():
    '''Function to retrieve the upper and lower retirement thresholds from the config file
    '''
    p_real = Config().thresholds[1]
    p_bogus = Config().thresholds[0]
    return [p_real,p_bogus]
    
def prior_setting():
    '''Function to retrieve the score prior from the config file'''
    return Config().p0

def read_sqlite(dbfile):
    '''Function to read-in the database files'''
    with sqlite3.connect(dbfile) as dbcon:
        tables = list(read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
        out = {tbl : read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}
    return out

hard_sims_ids = [91832396,91832397] #Random id's as example

def trajectory_plot(path='./data/swap.db'):
    '''Function to plot the score trajectories from the SWAP database.'''
    print(f'Plotting data from: {path}')
    subject_id=np.array(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id'])
    subject_histories = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['history']
    subject_golds=list(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['gold_label'])
    retired_bool = np.array(pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['retired'])==1
    retired_list = subject_id[np.where(retired_bool)]
    users = pd.DataFrame.from_dict(read_sqlite(path)['users'])
    assert len(subject_id)==len(set(subject_id)) #Subjects shouldn't be duplicated within the database.
    # max_seen is set by subject with max number of classifications
    max_seen = 1
    subjects_final = []
    subjects_history_final=[]
    history_all_final=[]
    golds_final=[]
    #Iterating through each subject...
    for indx in tqdm(range(len(subject_id))):
        subject = subject_id[indx]
        subjects_final.append(subject)
        golds_final.append(subject_golds[indx])
        history_i=[]
        classification_time_i=[]
        history_all_i=[]
        history_list_i=eval(subject_histories[indx])
        #...finding the subject score at each classification:
        for i in range(len(history_list_i)):
            history_i.append(history_list_i[i][3])
            history_all_i.append(history_list_i[i])
        #Adding these histories to one big list:
        subjects_history_final.append(history_i)
        history_all_final.append(history_all_i)
        max_seen = np.max([max_seen,len(history_i)])

    p_real, p_bogus = thresholds_setting()
    prior_value = prior_setting()

    fig, ax = pl.subplots(figsize=(3,3), dpi=300)
    ax.tick_params(labelsize=5)

    color_test = 'gray';color_bogus = 'red';color_real = 'blue';color_hardsim = 'green';color_retired = 'k'
    colors = [color_bogus, color_real, color_test]

    linewidth_test = 0.5; linewidth_bogus = 1.5; linewidth_real = 1.5;linewidth_hardsim = 1.5;linewidth_retired = 0.5
    linewidths = [linewidth_bogus, linewidth_real, linewidth_test]

    alpha_test = 0.1; alpha_bogus = 0.3; alpha_real = 0.3; alpha_hardsim = 0.3;alpha_retired = 0.1
    alphas = [alpha_bogus, alpha_real, alpha_test]

    # axes and labels
    p_min = 10**-10
    p_max = 1
    
    ax.set_xlim(p_min, p_max)
    ax.set_xscale('log')
    ax.set_ylim(max_seen+1,0)
    #Adding guide-lines for key scores:
    ax.axvline(x=prior_setting(), color=color_test, linestyle='dotted')
    ax.axvline(x=p_bogus, color=color_bogus, linestyle='dotted')
    ax.axvline(x=p_real, color=color_real, linestyle='dotted')
    #Shading the regions in which subjects are retired:
    ax.fill_betweenx(x1=p_min,x2=p_bogus,y=[max_seen+1,Config().lower_retirement_limit],alpha=0.2,color='darkviolet')
    ax.fill_betweenx(x1=p_bogus,x2=p_max,y=[max_seen+1,Config().retirement_limit],alpha=0.2,color='darkviolet')
    ax.set_xlabel('Posterior Probability Pr(LENS|d)',fontsize=5)
    ax.set_ylabel('Number of Classifications',fontsize=5)
    sub_list=[]
    #Iterating through the subjects, adding trajectory lines:
    for j in range(0,len(subjects_final)):
        history = np.array(subjects_history_final[j])
        y = np.arange(len(history))
        if subjects_final[j] in retired_list:
          #ax.annotate(str(subjects_final[j]),(history[-1:],y[-1:]),size=4) #To add subject-id annotations
            ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,c='k')
            ax.plot(history, y, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],
                    linewidth=linewidth_retired)
        elif subjects_final[j] in hard_sims_ids:
            ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,c=color_hardsim)
            ax.plot(history, y, linestyle='-',alpha=alpha_hardsim,color=color_hardsim,
                    linewidth=linewidth_hardsim)
        else:
            ax.plot(history, y, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],
                    linewidth=linewidth_test)
            ax.scatter(history, y,marker='+',linewidth=1,color=colors[golds_final[j]],alpha=alpha_retired,s=0.5)
            ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,\
                       edgecolors=colors[golds_final[j]],facecolors=colors[golds_final[j]])
    if len(hard_sims_ids)>0: colors.append(color_hardsim);alphas.append(alpha_hardsim)
    if len(retired_list)>0: colors.append(color_retired);alphas.append(alpha_retired)
    patches = []
    for color, alpha, label in zip(colors, alphas, ['Training: Dud', 'Training: Easy sim', 'Test']+\
                                                   ['Training: Hard sim']*(len(hard_sims_ids)>0)+\
                                                   ['Retired']*(len(retired_list)>0)):
        patch = mpatches.Patch(color=color, alpha=alpha, label=label)
        patches.append(patch)
    ax.legend(handles=patches, loc='lower right', framealpha=1.0,prop={'size':5})
    fig.tight_layout()
    pl.show()

trajectory_plot('/PATH/TO/SWAP/DATABASE/FILE')
