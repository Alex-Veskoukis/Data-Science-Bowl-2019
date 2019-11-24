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


# =============================================================================
# train.head()
# train_labels.head()
# =============================================================================


def Manipulate_Data(train, train_labels):
    train_Assessments = train.loc[train.type=='Assessment']
    Assessments = pd.merge(train_Assessments, train_labels, how='left', on=['installation_id', 'title', 'game_session'])
    Assessments['Attempt'] = 0
    AssessementTitles = Assessments['title'].unique()
    AssessementTitles1 = [item for item in AssessementTitles if item not in ['Bird Measurer (Assessment)']]
    Assessments.loc[Assessments['event_code'].isin([4100]) & Assessments.title.isin(AssessementTitles1),'Attempt'] = 1
    Assessments.loc[Assessments['event_code'].isin([4110]) & Assessments.title.isin(['Bird Measurer (Assessment)']),'Attempt'] = 1
    Assessments.loc[Assessments['event_data'].str.contains('false') & Assessments['Attempt'] == 1 ,'IsAttemptSuccessful'] = 0
    Assessments.loc[Assessments['event_data'].str.contains('true') & Assessments['Attempt'] == 1 ,'IsAttemptSuccessful'] = 1
    Assessments['timestamp'] = pd.to_datetime(Assessments['timestamp'] , format="%Y-%m-%d %H:%M")
    Assessments['NumOfPastAttempts'] = Assessments.groupby(['installation_id','game_session', 'title'])['Attempt'].transform(np.cumsum) - 1
    Assessments['IsSuccessfull'] = Assessments.groupby(['installation_id','game_session', 'title'])['IsAttemptSuccessful'].transform(np.cumsum)
    Assessments= Assessments[Assessments.Attempt == 1]
    Assessments['AttemptNumber'] = Assessments.groupby(['installation_id','game_session']).cumcount()+1
    Model_Base = Assessments[['installation_id', 'game_session','AttemptNumber','IsSuccessfull','NumOfPastAttempts','game_time','event_count','title','timestamp','accuracy','accuracy_group']].drop_duplicates()
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
    Model_Base= Model_Base[['accuracy','AttemptNumber','game_time','event_count','title','PartOfDay']]
    Model_Base = pd.concat([Model_Base, pd.get_dummies(Model_Base['PartOfDay'])] ,axis=1)
    del Model_Base['PartOfDay']
    Model_Base = pd.concat([Model_Base, pd.get_dummies(Model_Base['title'])] ,axis=1)
    del Model_Base['title']
    return Model_Base


ModelBase = Manipulate_Data(train, train_labels)
ModelBase.to_csv('ModelBase.csv')
