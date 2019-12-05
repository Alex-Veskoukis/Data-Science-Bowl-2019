# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:20:53 2019
@author: uocvc
"""

import os
# directory = 'C:/Users/Alex/Desktop/data-science-bowl-2019/Data'
# os.chdir(directory)
import pandas as pd
import numpy as np
import auxiliary_functions as af

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
train_labels = pd.read_csv('Data/train_labels.csv')
specs = pd.read_csv('Data/specs.csv')


def create_features(data):
    trainTitles = data['title'].unique()
    trainTitles_sub = [item for item in trainTitles if item not in ['Bird Measurer (Assessment)']]
    AttemptIndicator = (data.type == 'Assessment') & \
                       ((data.event_code.isin([4100]) & data.title.isin(trainTitles_sub)) |
                        (data.event_code.isin([4110]) & data.title.isin(['Bird Measurer (Assessment)'])))
    data['Attempt'] = 0
    data.loc[AttemptIndicator, 'Attempt'] = 1
    SuccessfulAttemptIndicator = data['event_data'].str.contains('true') & AttemptIndicator
    data['IsAttemptSuccessful'] = 0
    data.loc[SuccessfulAttemptIndicator, 'IsAttemptSuccessful'] = 1
    data['timestamp'] = pd.to_datetime(data['timestamp'], format="%Y-%m-%d %H:%M")
    Inst_Group = data.groupby('installation_id')
    Inst_Game_Group = data.groupby(['installation_id', 'game_session'])
    data['Total_Game_Session_Time'] = Inst_Game_Group['game_time'].transform(np.max)
    data['Total_Game_Session_Events'] = Inst_Game_Group['event_count'].transform(np.max)
    data['Assessments_played_Counter'] = data[data.type == 'Assessment'].groupby('installation_id')[
        'game_session'].transform(
        lambda x: np.round(pd.factorize(x)[0] + 1))
    data = data.sort_values(['installation_id', 'timestamp', 'game_session'], ascending=[True, True, True])
    data['Cumulative_Attempts'] = Inst_Group['Attempt'].transform(np.cumsum)
    data['Cumulative_Successes'] = Inst_Group['IsAttemptSuccessful'].transform(np.nancumsum)

    data['Assessment_Session_Time'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])[
        'game_time'].transform(np.max)
    data['Assessment_NumberOfEvents'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])[
        'event_count'].transform(np.max)
    # Slice 1
    slice1 = data.copy().loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type', 'title', 'world', 'Total_Game_Session_Time',
                              'Total_Game_Session_Events']].drop_duplicates()
    slice1['Game_Session_Order'] = slice1.groupby('installation_id')['game_session'].cumcount() + 1
    slice1['Cumulative_Time_Spent'] = slice1.groupby(['installation_id'])['Total_Game_Session_Time'].cumsum()
    # Slice 2
    slice2 = data.copy().loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type',
                              'Cumulative_Attempts', 'Cumulative_Successes',
                              'Assessments_played_Counter']].drop_duplicates()
    slice2['Game_Session_Order'] = slice2.groupby('installation_id')['game_session'].cumcount() + 1
    slice2 = slice2.sort_values(['installation_id', 'Game_Session_Order'])
    slice2 = slice2[slice2.type == 'Assessment']
    slice2['Past_Total_Attempts'] = round(
        slice2.groupby('installation_id')['Cumulative_Attempts'].shift(1, fill_value=0))
    slice2['Past_Total_Successes'] = round(
        slice2.groupby('installation_id')['Cumulative_Successes'].shift(1, fill_value=0))
    slice2['Past_Assessments_Played'] = round(
        slice2.groupby('installation_id')['Assessments_played_Counter'].shift(1, fill_value=0))
    slice2 = slice2.loc[:, ['installation_id', 'game_session',
                            'Game_Session_Order', 'Past_Total_Attempts',
                            'Past_Total_Successes', 'Past_Assessments_Played']]
    # Slice 3
    slice3 = data.loc[data.type == 'Assessment', ['installation_id',
                                                  'game_session',
                                                  'Assessment_Session_Time',
                                                  'Assessment_NumberOfEvents']].drop_duplicates()
    # Slice 1 / Type frequency Experience Measures
    type_slice = pd.pivot_table(slice1[['installation_id', 'game_session', 'type', 'Game_Session_Order']],
                                index=['installation_id', 'game_session', 'Game_Session_Order'],
                                columns='type',
                                aggfunc=len,
                                fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    type_slice['Past_Activities'] = type_slice.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice['Past_Games'] = type_slice.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice['Past_Clips'] = type_slice.groupby('installation_id')['Clip'].transform(np.cumsum)
    type_slice['Past_Assessments'] = type_slice.groupby('installation_id')['Assessment'].transform(np.cumsum)
    type_slice_assessments = type_slice[type_slice.Assessment == 1]
    type_slice_assessments = type_slice_assessments.rename(columns={'Game_Session_Order': 'Total_Game_Sessions'})
    type_slice_assessments = type_slice_assessments.drop(['Game', 'Clip', 'Assessment', 'Activity'], axis=1)
    type_slice_assessments['Clips'] = type_slice_assessments['Past_Clips']
    type_slice_assessments['Games'] = type_slice_assessments['Past_Games']
    type_slice_assessments['Assessments'] = type_slice_assessments['Past_Assessments']
    type_slice_assessments['Activities'] = type_slice_assessments['Past_Activities']
    type_slice_assessments = type_slice_assessments.loc[:,
                             ['installation_id', 'game_session', 'Total_Game_Sessions', 'Clips',
                              'Games', 'Assessments', 'Activities']].drop_duplicates()
    # Slice 1 / Type time spent Experience Measures
    type_slice2 = pd.pivot_table(
        slice1[['installation_id', 'game_session', 'type', 'Game_Session_Order', 'Total_Game_Session_Time']],
        index=['installation_id', 'game_session', 'Game_Session_Order'],
        columns='type',
        values='Total_Game_Session_Time',
        aggfunc=sum,
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    type_slice2['Time_spent_on_Activities'] = type_slice2.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice2['Time_spent_on_Games'] = type_slice2.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice2['Time_spent_on_Clips'] = type_slice2.groupby('installation_id')['Clip'].transform(np.cumsum)
    type_slice2['Time_spent_on_Assessments'] = type_slice2.groupby('installation_id')['Assessment'].transform(np.cumsum)
    type_slice2_assessments = type_slice2[type_slice2.Assessment != 0]
    type_slice2_assessments.loc[:, 'Total_Time_spent'] = type_slice2_assessments[
        ['Time_spent_on_Activities', 'Time_spent_on_Games', 'Time_spent_on_Clips', 'Time_spent_on_Assessments']].sum(
        axis=1)
    type_slice2_assessments = type_slice2_assessments.loc[:,
                              ['installation_id', 'game_session', 'Total_Time_spent', 'Time_spent_on_Activities',
                               'Time_spent_on_Games', 'Time_spent_on_Clips',
                               'Time_spent_on_Assessments']].drop_duplicates()

    MergedSlices = pd.merge(type_slice_assessments, type_slice2_assessments, on=['installation_id', 'game_session'],
                            how='inner')
    # Create Dummies
    Assessments = data[data.type == 'Assessment'].copy()
    Assessments['timestamp'] = pd.to_datetime(Assessments['timestamp'], format="%Y-%m-%d %H:%M")
    Assessments = Assessments.sort_values('timestamp', ascending=True)
    Assessments = Assessments.drop_duplicates()
    Assessments = af.convert_datetime(Assessments)
    Assessments['time'] = Assessments['timestamp'].dt.strftime('%H:%M')
    Assessments['date'] = Assessments['timestamp'].dt.date
    del Assessments['timestamp']
    Assessments['title'] = Assessments['title'].str.rstrip(' (Assessment)')
    Assessments = Assessments.set_index(['installation_id', 'game_session'])
    Assessments = Assessments[['title', 'world', 'month', 'hour', 'year', 'dayofweek']]
    Assessments['title'] = pd.Categorical(Assessments['title'])
    Assessments['world'] = pd.Categorical(Assessments['world'])
    Assessments = pd.concat([Assessments, pd.get_dummies(Assessments['world'])], axis=1)
    del Assessments['world']
    Assessments = pd.concat([Assessments, pd.get_dummies(Assessments['title'])], axis=1)
    del Assessments['title']
    Assessments = Assessments.reset_index()
    Assessments = Assessments.drop_duplicates()
    FinalData = pd.merge(Assessments, MergedSlices, how='inner',
                         on=['installation_id', 'game_session'])
    FinalData = pd.merge(FinalData, slice2, how='inner',
                         on=['installation_id', 'game_session'])
    FinalData = pd.merge(FinalData, slice3, how='inner',
                         on=['installation_id', 'game_session'])
    del FinalData['Game_Session_Order']
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
    Assess['accuracy_group'] = np.select(conditions, choices, default='black')
    Assess['accuracy'] = ratio
    Assess = Assess.reset_index()
    Assess = Assess.sort_values(['installation_id', 'timestamp'])
    Assess = Assess.groupby(['installation_id']).tail(1)
    return Assess[['accuracy', 'accuracy_group', 'installation_id', 'game_session']]


Test_set = get_test_set_accuracy(test)

Test_set_full = pd.merge(Test_Features, Test_set.loc[:, ~ Test_set.columns.isin(['accuracy'])],
                         on=['installation_id', 'game_session'], how='inner')

# Create Test and Control sets
X_train = FinalTrain.loc[:, ~FinalTrain.columns.isin(['accuracy_group', 'installation_id', 'game_session', 'PartOfDay'])]
Y_train = FinalTrain['accuracy_group'].astype(int)

X_test = Test_set_full.loc[:, ~Test_Features.columns.isin(['accuracy_group', 'installation_id', 'game_session'])]
Y_test = Test_set_full['accuracy_group'].to_numpy(dtype=int)


# To Test
# specs_unsplit = pd.DataFrame()
# import json
# for i in range(0, data.shape[0]):
#     for j in json.loads(data.event_data[i]):
#         new_df = pd.DataFrame({'event_id': data['event_data'][i]}, index=[i])
#         specs_unsplit = specs_unsplit.append(new_df)
#
# from sklearn.feature_extraction.text import CountVectorizer
#
# vec = CountVectorizer()
# sample = specs_unsplit['info']
# X = vec.fit_transform(sample)
# terms_count = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
#
# from sklearn.decomposition import PCA
# model = PCA(n_components=5)
# model.fit(terms_count)
# X_5D = model.transform(terms_count)
#
# reduced_terms = pd.DataFrame()
# reduced_terms['PCA1'] = X_5D[:, 0]
# reduced_terms['PCA2'] = X_5D[:, 1]
# reduced_terms['PCA3'] = X_5D[:, 2]
# reduced_terms['PCA4'] = X_5D[:, 3]
# reduced_terms['PCA5'] = X_5D[:, 4]
#
# # reduced_terms = pd.concat([reduced_terms.reset_index(drop=True),
# #                            specs_unsplit[['event_id']].reset_index(drop=True)], axis=1)
# # reduced_terms = reduced_terms.groupby('event_id')['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'].mean().reset_index()
# # # merge specs to train and test data
# data_spec = pd.merge(data[['installation_id', 'game_session','event_id']], reduced_terms, on='event_id', how='inner')
# data_compon = data_spec.groupby(['installation_id', 'game_session'])[
#     'PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5'].mean().reset_index()
# FinalData = pd.merge(FinalData, data_compon, how='inner',
#                      on=['installation_id', 'game_session'])
