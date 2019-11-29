# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:20:53 2019
@author: uocvc
"""

# =============================================================================
import os
# directory = 'C:/Users/Alex/Desktop/data-science-bowl-2019/Data'
# os.chdir(directory)
# =============================================================================
import pandas as pd
import numpy as np

train = pd.read_csv('Data/train.csv')
# =============================================================================
# train_labels = pd.read_csv('train_labels.csv')
# specs = pd.read_csv('specs.csv')
# test = pd.read_csv('test.csv')
# =============================================================================

Unique_Installations = train.installation_id.unique()
train = train[train.installation_id.isin(Unique_Installations[1:10])]
train['Attempt'] = 0
trainTitles = train['title'].unique()
trainTitles_sub = [item for item in trainTitles if item not in ['Bird Measurer (Assessment)']]
train.loc[train['event_code'].isin([4100]) & train.title.isin(trainTitles_sub), 'Attempt'] = 1
train.loc[train['event_code'].isin([4110]) & train.title.isin(['Bird Measurer (Assessment)']), 'Attempt'] = 1
train.loc[train['event_data'].str.contains('false') & train['Attempt'] == 1, 'IsAttemptSuccessful'] = 0
train.loc[train['event_data'].str.contains('true') & train['Attempt'] == 1, 'IsAttemptSuccessful'] = 1
train['timestamp'] = pd.to_datetime(train['timestamp'], format="%Y-%m-%d %H:%M")
train['Total_Game_Session_Time'] = train.groupby(['installation_id', 'game_session'])['game_time'].transform(np.max)
train['Total_Game_Session_Events'] = train.groupby(['installation_id', 'game_session'])['event_count'].transform(np.max)
Column_Order_List = [5, 3, 8, 2, 1, 10, 9, 11, 7, 6, 12, 13, 14, 15, 4]
train = train[[train.columns[i - 1] for i in Column_Order_List]]
train = train.sort_values(['installation_id', 'timestamp', 'game_session'], ascending=[True, True, True])
slice1 = train.loc[(train.game_time == train.Total_Game_Session_Time) &
                   (train.event_count == train.Total_Game_Session_Events),
                   ['installation_id', 'game_session', 'type', 'title', 'world', 'Total_Game_Session_Time',
                    'Total_Game_Session_Events']].drop_duplicates()
slice1['Order'] = slice1.groupby('installation_id')['game_session'].cumcount() + 1


# Type Experience Measures

type_slice = pd.pivot_table(slice1[['installation_id', 'game_session', 'type', 'Order']],
                            index=['installation_id', 'game_session', 'Order'],
                            columns='type',
                            aggfunc=len,
                            fill_value=0).reset_index().sort_values(['installation_id', 'Order'])
type_slice['Past_Activities'] = type_slice.groupby('installation_id')['Activity'].transform(np.cumsum)
type_slice['Past_Games'] = type_slice.groupby('installation_id')['Game'].transform(np.cumsum)
type_slice['Past_Clips'] = type_slice.groupby('installation_id')['Clip'].transform(np.cumsum)
type_slice['Past_Assessments'] = type_slice.groupby('installation_id')['Assessment'].transform(np.cumsum) - 1

type_slice_assessments = type_slice[type_slice.Assessment == 1]

type_slice_assessments = type_slice_assessments.rename(columns={'Order': 'Past_Game_Sessions'})
type_slice_assessments = type_slice_assessments.drop(['Game','Clip', 'Assessment','Activity'], axis = 1)


type_slice_assessments['Clip_Experience'] = type_slice_assessments['Past_Clips'] / type_slice_assessments['Past_Game_Sessions']
type_slice_assessments['Game_Experience'] = type_slice_assessments['Past_Games'] / type_slice_assessments['Past_Game_Sessions']
type_slice_assessments['Assessment_Experience'] = type_slice_assessments['Past_Assessments'] / type_slice_assessments['Past_Game_Sessions']
type_slice_assessments['Activity_Experience'] = type_slice_assessments['Past_Activities'] / type_slice_assessments['Past_Game_Sessions']

type_slice_assessments = type_slice_assessments.loc[:, ['installation_id', 'game_session', 'Clip_Experience', 'Game_Experience', 'Assessment_Experience', 'Activity_Experience']].drop_duplicates()