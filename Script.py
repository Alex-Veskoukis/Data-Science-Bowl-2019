# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 00:15:41 2019

@author: Alex
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:20:53 2019
@author: uocvc
"""


import os
directory = 'C:/Users/Alex/Desktop/data-science-bowl-2019/Data'
os.chdir(directory)
import pandas as pd
import numpy as np


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)



train = pd.read_csv('train.csv')
train_labels = pd.read_csv('train_labels.csv')
specs = pd.read_csv('specs.csv')
test = pd.read_csv('test.csv')


# =============================================================================
# train.head()
# train_labels.head()
# =============================================================================


def Manipulate_Baselile_Data(train):
    Assessments = train[train.type=='Assessment'].copy()
    Assessments['Attempt'] = 0
    AssessementTitles = Assessments['title'].unique()
    AssessementTitles1 = [item for item in AssessementTitles if item not in ['Bird Measurer (Assessment)']]
    Assessments.loc[Assessments['event_code'].isin([4100]) & Assessments.title.isin(AssessementTitles1),'Attempt'] = 1
    Assessments.loc[Assessments['event_code'].isin([4110]) & Assessments.title.isin(['Bird Measurer (Assessment)']),'Attempt'] = 1
    Assessments.loc[Assessments['event_data'].str.contains('false') & Assessments['Attempt'] == 1 ,'IsAttemptSuccessful'] = 0
    Assessments.loc[Assessments['event_data'].str.contains('true') & Assessments['Attempt'] == 1 ,'IsAttemptSuccessful'] = 1
    Assessments['timestamp'] = pd.to_datetime(Assessments['timestamp'] , format="%Y-%m-%d %H:%M")
    
    #Assessments['MeanEventTime'] = Assessments.groupby(['installation_id','event_id','game_session'])['game_time'].transform(np.mean)
    #Assessments['MaxEventTime'] = Assessments.groupby(['installation_id','event_id','game_session'])['game_time'].transform(np.max)
    Assessments['GameSessionTime'] = Assessments.groupby(['installation_id','game_session'])['game_time'].transform(np.max)
    Assessments['GameEventsCount'] = Assessments.groupby(['installation_id','game_session'])['event_count'].transform(np.max)
    Assessments = Assessments.sort_values('timestamp')
    Assessments= Assessments[Assessments.Attempt == 1]
    Model_Base = Assessments.drop_duplicates()
    Model_Base['time'] = Model_Base['timestamp'].dt.strftime('%H:%M')
    Model_Base['date'] = Model_Base['timestamp'].dt.date
    del Model_Base['timestamp']
    Model_Base.loc[Model_Base.time.between('05:01', '12:00', inclusive = True) ,"PartOfDay"] = 'Morning'
    Model_Base.loc[Model_Base.time.between('12:01', '17:00', inclusive = True) ,"PartOfDay"] = 'Afternoon'
    Model_Base.loc[Model_Base.time.between('17:01', '21:00', inclusive = True) ,"PartOfDay"] = 'Evening'
    Model_Base.loc[Model_Base.time.between('21:01', '23:59', inclusive = True) ,"PartOfDay"] = 'Night'
    Model_Base.loc[Model_Base.time.between('00:00', '05:00', inclusive = True) ,"PartOfDay"] = 'Night'
    Model_Base['title'] = Model_Base['title'].str.rstrip(' (Assessment)')
    Model_Base = Model_Base.set_index(['installation_id','game_session'])
    Model_Base= Model_Base[['title','PartOfDay','world','GameSessionTime','GameEventsCount']]
    Model_Base = pd.concat([Model_Base, pd.get_dummies(Model_Base['PartOfDay'])] ,axis=1)
    del Model_Base['PartOfDay']
    Model_Base = pd.concat([Model_Base, pd.get_dummies(Model_Base['world'])] ,axis=1)
    del Model_Base['world']
    #Model_Base['NumOfPreviousGames'] = Assessments.groupby(['installation_id'])['game_session'].transform('count')
    return Model_Base.drop_duplicates()



ModelBase = Manipulate_Baselile_Data(train)

# Attach ground truth on train
train_labels['title'] = train_labels['title'].str.rstrip(' (Assessment)')
Accuracy = train_labels[['installation_id', 'title', 'game_session','accuracy']]
ModelBaseWithAccuracy = pd.merge(Accuracy, ModelBase,  how='right', on=['installation_id', 'title', 'game_session'])
ModelBaseWithAccuracy = pd.concat([ModelBaseWithAccuracy, pd.get_dummies(ModelBaseWithAccuracy['title'])] ,axis=1)
del ModelBaseWithAccuracy['title']
ModelBaseWithAccuracy.to_csv('ModelBaseWithAccuracy.csv')


ModelBaseTest = Manipulate_Baselile_Data(test)

