"""For plotting trajectories.
Adapted from https://github.com/drphilmarshall/SpaceWarps with minor adjustments

"""
def thresholds_setting():
#    p_real = 0.95
#    p_bogus = 1.e-7
    p_real = 0.95
    p_bogus = 1e-7
    return [p_real,p_bogus]
def prior_setting():
    return 5.e-4

def read_sqlite(dbfile):
    import sqlite3
    from pandas import read_sql_query, read_sql_table
    with sqlite3.connect(dbfile) as dbcon:
      tables = list(read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", dbcon)['name'])
      out = {tbl : read_sql_query(f"SELECT * from {tbl}", dbcon) for tbl in tables}
    return out

def trajectory_plot(path='./data/online_swap.db', subjects=200, logy=True):
    import pandas as pd
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    subject_id=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['subject_id']
    users =pd.DataFrame.from_dict(read_sqlite(path)['users'])
    subject_histories = pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['history']
    subject_golds=pd.DataFrame.from_dict(read_sqlite(path)['subjects'])['gold_label']
    # get subjects
    # max_seen is set by subject with max number of classifications
    max_seen = 1
    print('user_b:1517738')
    user_b_indx=list(users['user_id']).index(1517738)
    print('user_history')
    print(str(eval(users['history'][user_b_indx])[0:10])+'...')

    subjects=[20937590,20937591,20937592,20937593,20937595,20937596,20937599,20937601,20937603,20937605,20937594,20937597,20937604,20937609,20937600,20937611,20937612,20937614,20937615,20937602,20964743,20964703,20964731,20964726,20964727,20964739,20964711,20964735,20964701,20964705,20964697,20964702,20964707,20964720,20964723,20964724,20964730,20964738,20964745,20964746,20964700,20964704,20964717,20964718,20964732,20964747,20964709,20964715,20964719,20964725,20964729,20964733,20865730,20865735,20865731,21100285,21088876,21099617,21102901,21091587,21089681,21093839,21093663,21093235,21093000,21087994,21087602,21093448,21099911,21090710,21089950,21107573,21015283,21088837,21107551,21097925,21102933,21038842,21107039,21098397,21099179,21102560,21105527,21089824,21108230,21106003,21090396,21093977,21093201,21109050,21088007,21098611,21099829,21097483,21106327,21091737,21091986,21006238,21104125,21094176,21102839,21090693,21100556,21101657,21093582,21103010,21092401,21097506,21093610,21090539,21097274,21099283,21100519,21094684,21087524,21094479,21103251,21103610,21090300,21107223,21091209,21090600,21087537,21103845,21089490,21095359,21098539,21093315,21103957,21097302,21009414,21044528,21095314,21095649,21098715,21106457,21105992,21092409,21096841,21092173,21088193,21097954,21102761,21028279,21107577,21101093,21093955,21099147,21102456,21103836,21088653,21088758,21101575,21108270,21096475,21097284,21087623,21106114,21087326,21093222,21106768,21099849,21090466,21107939,21098730,21095088,21104057,21105411,21107705,21097753,21034160,21102709,21096422,21093766,21102879,21105049,21102686,21108077,21108948,21104113,21107678,21092098,21099087,21094327,21094775,21099047,21093891,21105394,21094341,21094307,21091318,21105831,21090278,21094673,20962148,21097341,21102510,21105817,21090989,21103796]
    subjects_final = []
    subjects_history_final=[]
    history_all_final=[]
    golds_final=[]
    if type(subjects) == int:
        # draw random numbers
        while len(subjects_final) < subjects:
            # note that there can be duplication.
            indx = np.random.choice(len(subject_id))
            subject = subject_id[indx]
            # if subject.seen < 3:
            #     continue
            subjects_final.append(subject)
            golds_final.append(subject_golds[indx])
            history_i=[]
            history_all_i=[]
            history_list_i=eval(subject_histories[indx])
            for i in range(len(history_list_i)):
              history_i.append(history_list_i[i][3])
              history_all_i.append(history_list_i[i])
            subjects_history_final.append(history_i)
            history_all_final.append(history_all_i)
            if len(history_i) > max_seen:
                max_seen = len(history_i)
    else:
      for p in range(len(subjects)):
        indx = p
        subjects_final.append(subjects[indx])
        golds_final.append(subject_golds[indx])
        history_i=[]
        history_all_i=[]
        history_list_i=eval(subject_histories[indx])
        for i in range(len(history_list_i)):
          history_i.append(history_list_i[i][3])
          history_all_i.append(history_list_i[i])
        subjects_history_final.append(history_i)
        history_all_final.append(history_all_i)
        if len(history_i) > max_seen:
            max_seen = len(history_i)
    subjects = subjects_final
    fig, ax = plt.subplots(figsize=(3,3), dpi=300)

    #####
    # pretty up the figure
    #####
    color_test = 'gray'
    color_bogus = 'red'
    color_real = 'blue'
    colors = [color_bogus, color_real, color_test]

    linewidth_test = 1.0
    linewidth_bogus = 1.5
    linewidth_real = 1.5
    linewidths = [linewidth_bogus, linewidth_real, linewidth_test]

    size_test = 20
    size_bogus = 40
    size_real = 40
    sizes = [size_bogus, size_real, size_test]

    alpha_test = 0.1
    alpha_bogus = 0.3
    alpha_real = 0.3
    alphas = [alpha_bogus, alpha_real, alpha_test]


    # axes and labels
    p_min = 5e-8
    p_max = 1
    ax.set_xlim(p_min, p_max)
    ax.set_xscale('log')
    ax.set_ylim(max_seen,1)
    if logy:
        ax.set_yscale('log')

    p_real , p_bogus= thresholds_setting()

    ax.axvline(x=prior_setting(), color=color_test, linestyle='dotted')
    ax.axvline(x=p_bogus, color=color_bogus, linestyle='dotted')
    ax.axvline(x=p_real, color=color_real, linestyle='dotted')
    ax.set_xlabel('Posterior Probability Pr(LENS|d)',fontsize=5)
    ax.set_ylabel('No. of Classifications',fontsize=5)
    ax.tick_params(labelsize=5)
    # plot history trajectories
    for j in range(len(subjects)):
        # clip history
        history = np.array(subjects_history_final[j])
        history = np.where(history < p_min, p_min, history)
        history = np.where(history > p_max, p_max, history)
        # trajectory
        y = np.arange(len(history) + 1)

        # add initial value
        history = np.append(prior_setting(), history)
        if subjects[j]==20937590:
          print(np.array(history))
          print(golds_final[j])
          print(history_all_final[j])
        ax.plot(history, y, linestyle='-',color=colors[golds_final[j]],alpha=alphas[golds_final[j]],linewidth=0.5)

        # a point at the end
        ax.scatter(history[-1:], y[-1:], alpha=1.0,s=1,edgecolors=colors[golds_final[j]],facecolors=colors[golds_final[j]])

    # add legend
    patches = []
    for color, alpha, label in zip(colors, alphas, ['Bogus', 'Real', 'Test']):
        patch = mpatches.Patch(color=color, alpha=alpha, label=label)
        patches.append(patch)
    ax.legend(handles=patches, loc='lower center', framealpha=1.0,prop={'size':5})

    fig.tight_layout()

#    if path:
#        fig.savefig(path, dpi=300)

    plt.show()
trajectory_plot()
