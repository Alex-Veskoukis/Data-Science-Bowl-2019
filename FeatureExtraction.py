# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:20:53 2019
@author: uocvc
"""


# =============================================================================
# import os
# directory = 'C:/Users/Alex/Desktop/data-science-bowl-2019/Data'
# os.chdir(directory)
# =============================================================================
import pandas as pd
import numpy as np



train = pd.read_csv('train.csv')
# =============================================================================
# train_labels = pd.read_csv('train_labels.csv')
# specs = pd.read_csv('specs.csv')
# test = pd.read_csv('test.csv')
# =============================================================================

Unique_Installations = train.installation_id.unique()
train = train[train.installation_id.isin(Unique_Installations[1:10])]
#train = train[train.type=='Assessment'].copy()
train['Attempt'] = 0
trainTitles = train['title'].unique()
trainTitles_sub = [item for item in trainTitles if item not in ['Bird Measurer (Assessment)']]
train.loc[train['event_code'].isin([4100]) & train.title.isin(trainTitles_sub),'Attempt'] = 1
train.loc[train['event_code'].isin([4110]) & train.title.isin(['Bird Measurer (Assessment)']),'Attempt'] = 1
train.loc[train['event_data'].str.contains('false') & train['Attempt'] == 1 ,'IsAttemptSuccessful'] = 0
train.loc[train['event_data'].str.contains('true') & train['Attempt'] == 1 ,'IsAttemptSuccessful'] = 1
train['timestamp'] = pd.to_datetime(train['timestamp'] , format="%Y-%m-%d %H:%M")

#Assessments['MeanEventTime'] = Assessments.groupby(['installation_id','event_id','game_session'])['game_time'].transform(np.mean)
#Assessments['MaxEventTime'] = Assessments.groupby(['installation_id','event_id','game_session'])['game_time'].transform(np.max)
train['Total_Game_Session_Time'] = train.groupby(['installation_id','game_session'])['game_time'].transform(np.max)
train['Total_Game_Session_Events'] = train.groupby(['installation_id','game_session'])['event_count'].transform(np.max)
#train = train.sort_values('timestamp', ascending = True)


Column_Order_List = [5,3,8,2,1,10,9,11,7,6,12,13,14,15,4] 
train = train[[train.columns[i-1] for i in Column_Order_List]]


train = train.sort_values(['installation_id','timestamp','game_session'],ascending = [True,True, True])


slicetrain = train.loc[(train.game_time == train.Total_Game_Session_Time) & (train.event_count == train.Total_Game_Session_Events),
                       ['installation_id','game_session','type','title','world',
                        'Total_Game_Session_Time','Total_Game_Session_Events']].drop_duplicates()

slicetrain['Order'] = slicetrain.groupby('installation_id')['game_session'].cumcount() + 1

typesclice = pd.pivot_table(slicetrain[['installation_id','game_session','type','Order']], 
                            index = ['installation_id','game_session','Order'], 
                            columns = 'type', 
                            aggfunc = len).reset_index().sort_values(['installation_id','Order'])

