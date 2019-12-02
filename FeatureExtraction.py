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
test = pd.read_csv('Data/test.csv')
# =============================================================================
train_labels = pd.read_csv('Data/train_labels.csv')


# specs = pd.read_csv('specs.csv')
# =============================================================================
#
# Unique_Installations = train.installation_id.unique()
# train = train[train.installation_id.isin(Unique_Installations[1:10])]

def create_features(data):
    data['Attempt'] = 0
    # trainTitles = data['title'].unique()
    # trainTitles_sub = [item for item in trainTitles if item not in ['Bird Measurer (Assessment)']]
    # data.loc[data['event_code'].isin([4100]) & data.title.isin(trainTitles_sub), 'Attempt'] = 1
    # data.loc[data['event_code'].isin([4110]) & data.title.isin(['Bird Measurer (Assessment)']), 'Attempt'] = 1
    # data.loc[data['event_data'].str.contains('false') & data['Attempt'] == 1, 'IsAttemptSuccessful'] = 0
    # data.loc[data['event_data'].str.contains('true') & data['Attempt'] == 1, 'IsAttemptSuccessful'] = 1
    data['timestamp'] = pd.to_datetime(data['timestamp'], format="%Y-%m-%d %H:%M")
    data['Total_Game_Session_Time'] = data.groupby(['installation_id', 'game_session'])['game_time'].transform(np.max)
    data['Total_Game_Session_Events'] = data.groupby(['installation_id', 'game_session'])['event_count'].transform(np.max)
    Column_Order_List = [5, 3, 8, 2, 1, 10, 9, 11, 7, 6, 12, 13, 14, 15, 4]
    data = data[[data.columns[i - 1] for i in Column_Order_List]]
    data = data.sort_values(['installation_id', 'timestamp', 'game_session'], ascending=[True, True, True])
    # Slice 1
    slice1 = data.loc[(data.game_time == data.Total_Game_Session_Time) &
                      (data.event_count == data.Total_Game_Session_Events),
                      ['installation_id', 'game_session', 'type', 'title', 'world', 'Total_Game_Session_Time',
                       'Total_Game_Session_Events']].drop_duplicates()
    slice1['Order'] = slice1.groupby('installation_id')['game_session'].cumcount() + 1
    slice1['Time_Spent'] = slice1.groupby(['installation_id'])['Total_Game_Session_Time'].cumsum()
    # Slice 1 / Type frequency Experience Measures
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
    type_slice_assessments = type_slice_assessments.drop(['Game', 'Clip', 'Assessment', 'Activity'], axis=1)
    # Past_Game_Sessions = type_slice_assessments['Past_Game_Sessions']
    # type_slice_assessments['Clip_Experience'] = type_slice_assessments['Past_Clips'] / Past_Game_Sessions
    # type_slice_assessments['Game_Experience'] = type_slice_assessments['Past_Games'] / Past_Game_Sessions
    # type_slice_assessments['Assessment_Experience'] = type_slice_assessments['Past_Assessments'] / Past_Game_Sessions
    # type_slice_assessments['Activity_Experience'] = type_slice_assessments['Past_Activities'] / Past_Game_Sessions
    type_slice_assessments['Clip_Experience'] = type_slice_assessments['Past_Clips']
    type_slice_assessments['Game_Experience'] = type_slice_assessments['Past_Games']
    type_slice_assessments['Assessment_Experience'] = type_slice_assessments['Past_Assessments']
    type_slice_assessments['Activity_Experience'] = type_slice_assessments['Past_Activities']
    type_slice_assessments = type_slice_assessments.loc[:,
                             ['installation_id', 'game_session','Past_Game_Sessions', 'Clip_Experience', 'Game_Experience',
                              'Assessment_Experience', 'Activity_Experience']].drop_duplicates()
    # Slice 1 / Type time spent Experience Measures
    type_slice2 = pd.pivot_table(
        slice1[['installation_id', 'game_session', 'type', 'Order', 'Total_Game_Session_Time']],
        index=['installation_id', 'game_session', 'Order'],
        columns='type',
        values='Total_Game_Session_Time',
        aggfunc=sum,
        fill_value=0).reset_index().sort_values(['installation_id', 'Order'])
    type_slice2['Past_Activities'] = type_slice2.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice2['Past_Games'] = type_slice2.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice2['Past_Clips'] = type_slice2.groupby('installation_id')['Clip'].transform(np.cumsum)
    type_slice2['Past_Assessments'] = type_slice2.groupby('installation_id')['Assessment'].transform(np.cumsum)
    type_slice2_assessments = type_slice2[type_slice2.Assessment != 0]
    type_slice2_assessments.loc[:, 'Past_Assessments'] = type_slice2_assessments.groupby('installation_id')[
        'Past_Assessments'].transform(lambda x: x.shift(1, fill_value=0))
    type_slice2_assessments.loc[:,'TotalPastTime'] = type_slice2_assessments[['Past_Activities','Past_Games','Past_Clips','Past_Assessments']].sum(axis=1)
    # type_slice2_assessments.loc[:, 'Clip_Time_Exp'] = type_slice2_assessments['Past_Clips'] / TotalPastTime
    # type_slice2_assessments.loc[:, 'Game_Time_Exp'] = type_slice2_assessments['Past_Games'] / TotalPastTime
    # type_slice2_assessments.loc[:, 'Assessment_Time_Exp'] = type_slice2_assessments['Past_Assessments'] / TotalPastTime
    # type_slice2_assessments.loc[:, 'Activity_Time_Exp'] = type_slice2_assessments['Past_Activities'] / TotalPastTime
    type_slice2_assessments.loc[:,'Clip_Time_Exp'] = type_slice2_assessments['Past_Clips']
    type_slice2_assessments.loc[:,'Game_Time_Exp'] = type_slice2_assessments['Past_Games']
    type_slice2_assessments.loc[:,'Assessment_Time_Exp'] = type_slice2_assessments['Past_Assessments']
    type_slice2_assessments.loc[:,'Activity_Time_Exp'] = type_slice2_assessments['Past_Activities']
    type_slice2_assessments = type_slice2_assessments.loc[:,
                              ['installation_id', 'game_session', 'TotalPastTime','Game_Time_Exp',
                               'Assessment_Time_Exp', 'Activity_Time_Exp']].drop_duplicates()
    MergedSlices = pd.merge(type_slice_assessments, type_slice2_assessments, on=['installation_id', 'game_session'],
                            how='inner')
    # Create Dummies
    Assessments = data[data.type == 'Assessment'].copy()
    Assessments['timestamp'] = pd.to_datetime(Assessments['timestamp'], format="%Y-%m-%d %H:%M")
    Assessments = Assessments.sort_values('timestamp', ascending=True)
    Assessments = Assessments.drop_duplicates()
    Assessments['time'] = Assessments['timestamp'].dt.strftime('%H:%M')
    Assessments['date'] = Assessments['timestamp'].dt.date
    del Assessments['timestamp']
    Assessments.loc[Assessments.time.between('05:01', '12:00', inclusive=True), "PartOfDay"] = 'Morning'
    Assessments.loc[Assessments.time.between('12:01', '17:00', inclusive=True), "PartOfDay"] = 'Afternoon'
    Assessments.loc[Assessments.time.between('17:01', '21:00', inclusive=True), "PartOfDay"] = 'Evening'
    Assessments.loc[Assessments.time.between('21:01', '23:59', inclusive=True), "PartOfDay"] = 'Night'
    Assessments.loc[Assessments.time.between('00:00', '05:00', inclusive=True), "PartOfDay"] = 'Night'
    Assessments['title'] = Assessments['title'].str.rstrip(' (Assessment)')
    Assessments = Assessments.set_index(['installation_id', 'game_session'])
    Assessments = Assessments[['title', 'PartOfDay', 'world']]
    Assessments['title'] = pd.Categorical(Assessments['title'])
    # Assessments['title'] = Assessments['title'].cat.codes
    Assessments['PartOfDay'] =pd.Categorical(Assessments['PartOfDay'])
    # Assessments['PartOfDay'] = Assessments['PartOfDay'].cat.codes
    Assessments['world'] = pd.Categorical(Assessments['world'])
    # Assessments['world'] = Assessments['world'].cat.codes
    Assessments = pd.concat([Assessments, pd.get_dummies(Assessments['PartOfDay'])], axis=1)
    del Assessments['PartOfDay']
    Assessments = pd.concat([Assessments, pd.get_dummies(Assessments['world'])], axis=1)
    del Assessments['world']
    Assessments = pd.concat([Assessments, pd.get_dummies(Assessments['title'])], axis=1)
    Assessments = Assessments.drop(['title'], axis=1)
    Assessments = Assessments.reset_index()
    Assessments = Assessments.drop_duplicates()
    FinalData = pd.merge(Assessments, MergedSlices, how='inner',
                         on=['installation_id', 'game_session'])
    return FinalData


Final = create_features(train)

FinalTrain = pd.merge(Final,
                      train_labels[['installation_id', 'game_session', 'accuracy_group']],
                      how='inner',
                      on=['installation_id', 'game_session'])

FinalTrain = FinalTrain.set_index(['installation_id', 'game_session'])

Test_Features = create_features(test)


def get_test_set_accuracy(data):
    Assess = data[data.type == 'Assessment'].copy()
    Assess['Attempt'] = 0
    AssessmentTitles = Assess['title'].unique()
    AssessmentTitles1 = [item for item in AssessmentTitles if item not in ['Bird Measurer (Assessment)']]
    Assess.loc[Assess['event_code'].isin([4100]) & Assess.title.isin(AssessmentTitles1), 'Attempt'] = 1
    Assess.loc[
        Assess['event_code'].isin([4110]) & Assess.title.isin(['Bird Measurer (Assessment)']), 'Attempt'] = 1
    Assess.loc[
        Assess['event_data'].str.contains('false') & Assess['Attempt'] == 1, 'IsAttemptSuccessful'] = 0
    Assess.loc[
        Assess['event_data'].str.contains('true') & Assess['Attempt'] == 1, 'IsAttemptSuccessful'] = 1
    Assess['timestamp'] = pd.to_datetime(Assess['timestamp'], format="%Y-%m-%d %H:%M")
    Assess = Assess[Assess.Attempt == 1]
    Assess = Assess.drop_duplicates()
    Assess['PastAssessmentGames'] = Assess.groupby('installation_id')['game_session'].transform(
        lambda x: pd.factorize(x)[0])
    Assess = Assess.sort_values(['installation_id', 'timestamp'], ascending=[True, True])
    Assess['Attempts'] = Assess.groupby(['installation_id', 'game_session'])['Attempt'].transform(np.sum)
    Assess['Success'] = Assess.groupby(['installation_id', 'game_session'])['IsAttemptSuccessful'].transform(
        np.sum)
    Assess = Assess.set_index(['installation_id', 'game_session'])
    Assess = Assess[['Attempts', 'Success', 'timestamp']]
    ratio = Assess['Success'] / Assess['Attempts']
    conditions = [
        (ratio == 1),
        (ratio == 0.5),
        (ratio < 0.5) & (ratio > 0),
        (ratio == 0)]
    choices = [3, 2, 1, 0]
    # =============================================================================
    # ModelBase_Test['accuracy_group'] = [3 if   (v == 1)
    #                                       else 2 if v  == 0.5
    #                                       else 1 if  (v < 0.5 ) & (v  > 0 )
    #                                       else 0  for v in ratio ]
    # =============================================================================
    Assess['accuracy_group'] = np.select(conditions, choices, default='black')
    Assess['accuracy'] = ratio
    Assess = Assess.reset_index()
    Assess = Assess.sort_values(['installation_id', 'timestamp'])
    Assess = Assess.groupby(['installation_id']).tail(1)
    return Assess[['accuracy','accuracy_group', 'installation_id', 'game_session']]


Test_set = get_test_set_accuracy(test)

Test_set_full = pd.merge(Test_Features, Test_set.loc[:, ~ Test_set.columns.isin(['accuracy'])], on=['installation_id', 'game_session'], how='inner')


X_train = FinalTrain.loc[:, ~FinalTrain.columns.isin(['accuracy_group','installation_id', 'game_session','PartOfDay'])]
Y_train = FinalTrain['accuracy_group']


X_test = Test_set_full.loc[:, ~Test_Features.columns.isin(['accuracy_group','installation_id', 'game_session'])]
Y_test = Test_set_full['accuracy_group'].to_numpy(dtype=int)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

import auxiliary_functions

quadratic_weighted_kappa(Y_test, Y_pred)


from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

quadratic_weighted_kappa(Y_test, Y_pred)

