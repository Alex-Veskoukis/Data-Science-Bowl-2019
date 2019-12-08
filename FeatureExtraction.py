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
from functools import reduce

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
train_labels = pd.read_csv('Data/train_labels.csv')
specs = pd.read_csv('Data/specs.csv')


def create_features(data):
    global col1
    global col2
    global col3
    global col4
    global col5
    global col6
    global col7
    global col8
    global col9
    global col10
    global col11
    global col12
    global col13
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
    data = data.sort_values(['installation_id', 'timestamp', 'game_session'], ascending=[True, True, True])
    Inst_Group = data.groupby('installation_id')
    Inst_Game_Group = data.groupby(['installation_id', 'game_session'])
    # initial measures
    data['Total_Game_Session_Time'] = Inst_Game_Group['game_time'].transform(np.max)
    data['Total_Game_Session_Events'] = Inst_Game_Group['event_count'].transform(np.max)
    data['Assessments_played_Counter'] = data[data.type == 'Assessment'].groupby('installation_id')[
        'game_session'].transform(lambda x: np.round(pd.factorize(x)[0] + 1))
    data['Cumulative_Attempts'] = Inst_Group['Attempt'].transform(np.cumsum)
    data['Cumulative_Successes'] = Inst_Group['IsAttemptSuccessful'].transform(np.nancumsum)

    data['Assessment_Session_Time'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])[
        'game_time'].transform(np.max)
    data['Assessment_NumberOfEvents'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])[
        'event_count'].transform(np.max)
    # Previous Accuracy
    previous_accuracy_metrics = af.get_previous_ac_metrics(data)
    # Slice 1
    slice1 = data.copy().loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type', 'title', 'world', 'Total_Game_Session_Time',
                              'Total_Game_Session_Events']].drop_duplicates()
    slice1['Game_Session_Order'] = slice1.groupby('installation_id')['game_session'].cumcount() + 1
    slice1['Cumulative_Time_Spent'] = slice1.groupby(['installation_id'])['Total_Game_Session_Time'].cumsum()
    # Slice 2
    Number_of_attemps_and_successes = af.get_past_attemps_and_successes(data)
    # Slice 3
    past_assessment_time_events_and_metrics = af.get_past_assessment_time_events_and_metrics(data)
    # Event_and_Attempts
    pre_time_till_attempt_metrics = af.get_prev_events_and_time_till_attempt(data)
    # title dummies
    title_visits = af.get_vists_per_title(slice1)
    # Slice 1 / titles times
    cummulative_time_spent_in_titles = af.get_cummulative_time_spent_in_titles(slice1)
    # Slice 1 / events count
    cummulative_events_seen_per_title = af.get_cummulative_events_seen_per_title(slice1)
    # Slice 8
    slice8 = data.loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type',
                              'title',
                              'Assessments_played_Counter',
                              'Cumulative_Attempts',
                              'Cumulative_Successes'
                              ]].copy().drop_duplicates()

    slice8['Game_Session_Order'] = slice8.groupby('installation_id')['game_session'].cumcount() + 1
    slice8 = slice8.sort_values(['installation_id', 'Game_Session_Order'])
    slice8['Past_Total_Attempts'] = round(
        slice8.groupby('installation_id')['Cumulative_Attempts'].shift(1, fill_value=0))
    slice8['Past_Total_Successes'] = round(
        slice8.groupby('installation_id')['Cumulative_Successes'].shift(1, fill_value=0))
    slice8['Past_Assessments_Played'] = round(
        slice8.groupby('installation_id')['Assessments_played_Counter'].shift(1, fill_value=0))

    slice8['Game_Session_Order'] = slice8.groupby('installation_id')['game_session'].cumcount() + 1
    cummulative_attempts_per_title = af.get_cummulative_attempts_per_title(slice8)
    # Slice 9
    cummulative_successes_per_title = af.get_cummulative_successes_per_title(slice8)
    # Slice 10
    cummulative_past_assesments_per_title = af.get_cummulative_past_assesments_per_title(slice8)
    # Slice 1 / Type frequency Experience Measures
    Number_of_games_played_per_type = af.get_frequency_per_type(slice1)
    # Slice 1 / Type time spent Experience Measures
    Time_spent_on_games_metrics = af.get_cumulative_time_spent_on_types(slice1)
    # Slice 1 / world time spent Experience Measures
    time_spent_on_diffrent_worlds = af.get_time_spent_on_diffrent_worlds(slice1)
    # Substract Level of Player
    Level_reached = af.substract_level(slice1)
    # Create Dummies
    world_time_gametitles_dummies = af.create_world_time_assesstitle_Dummies(data)
    # Get all together
    Sets = [Number_of_games_played_per_type,
            Time_spent_on_games_metrics,
            world_time_gametitles_dummies,
            Number_of_attemps_and_successes,
            past_assessment_time_events_and_metrics,
            pre_time_till_attempt_metrics,
            time_spent_on_diffrent_worlds,
            Level_reached,
            previous_accuracy_metrics,
            title_visits,
            cummulative_time_spent_in_titles,
            cummulative_events_seen_per_title,
            cummulative_attempts_per_title,
            cummulative_successes_per_title,
            cummulative_past_assesments_per_title]
    FinalData = reduce(lambda left, right: pd.merge(left, right, how='inner',on=['installation_id', 'game_session']), Sets)

    col1 = Number_of_games_played_per_type.columns
    col2 = Time_spent_on_games_metrics.columns
    col3 = past_assessment_time_events_and_metrics.columns
    col4 = pre_time_till_attempt_metrics.columns
    col5 = time_spent_on_diffrent_worlds.columns
    col6 = Level_reached.columns
    col7 = previous_accuracy_metrics.columns
    col8 = title_visits.columns
    col9 = cummulative_time_spent_in_titles.columns
    col10 = cummulative_events_seen_per_title.columns
    col11 = cummulative_attempts_per_title.columns
    col12 = cummulative_successes_per_title.columns
    col13 = cummulative_past_assesments_per_title.columns
    return FinalData


Final = create_features(train)

FinalTrain = pd.merge(Final,
                      train_labels[['installation_id', 'game_session', 'accuracy_group']],
                      how='inner',
                      on=['installation_id', 'game_session'])

FinalTrain = FinalTrain.set_index(['installation_id', 'game_session'])

Test_Features = create_features(test)

# '00abaee7' '348d7f09f96af313
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
    Assess['Attempts'] = Assess.groupby(['installation_id', 'game_session'])['Attempt'].transform(np.sum)
    Assess['Success'] = Assess.groupby(['installation_id', 'game_session'])['IsAttemptSuccessful'].transform(np.sum)
    Assess = Assess.set_index(['installation_id', 'game_session'])
    Assess = Assess[['Attempts', 'Success', 'timestamp']]
    ratio = Assess['Success'] / Assess['Attempts']
    conditions = [
        (ratio == 1),
        (ratio == 0.5),
        (ratio < 0.5) & (ratio > 0),
        (ratio == 0)]
    choices = [3, 2, 1, 0]
    Assess['accuracy_group'] = np.select(conditions, choices)
    Assess['accuracy'] = ratio
    Assess = Assess.reset_index()
    Assess = Assess.sort_values(['installation_id', 'timestamp'])
    Assess = Assess[[ 'accuracy', 'accuracy_group', 'installation_id', 'game_session']]
    Assess["To_Predict"] = 0
    Assess['order']=Assess.groupby('installation_id')[
        'game_session'].transform(lambda x: np.round(pd.factorize(x)[0] + 1))
    Assess['LastGame'] = Assess.groupby('installation_id')['order'].transform('max')
    Assess.loc[Assess.order == Assess.LastGame, "To_Predict"] = 1
    Assess = Assess.drop('accuracy', axis=1)
    Assess = Assess.drop_duplicates()
    return Assess


Test_set = get_test_set_accuracy(test)

Test_set_full = pd.merge(Test_Features, Test_set.loc[:, ~ Test_set.columns.isin(['accuracy'])],
                         on=['installation_id', 'game_session'], how='inner')
Test_set_full = Test_set_full[Test_set_full.To_Predict == 1]
# Create Test and Control sets
X_train = FinalTrain.loc[:,
          ~FinalTrain.columns.isin(['accuracy_group', 'installation_id', 'game_session'])]
Y_train = FinalTrain['accuracy_group'].astype(int)

X_test = Test_set_full.loc[:, ~Test_Features.columns.isin(['accuracy_group', 'installation_id', 'game_session', 'To_Predict', 'order', 'LastGame'])]
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
