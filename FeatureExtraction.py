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
    data = data.sort_values(['installation_id', 'timestamp', 'game_session'], ascending=[True, True, True])
    Inst_Group = data.groupby('installation_id')
    Inst_Game_Group = data.groupby(['installation_id', 'game_session'])
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
    Assess = data.sort_values(['installation_id', 'timestamp', 'game_session'], ascending=[True, True, True]).copy()
    Assess = Assess[Assess.type == 'Assessment']
    Assess['Attempts'] = Assess.groupby(['installation_id', 'game_session'])['Attempt'].transform(np.sum)
    Assess['Success'] = Assess.groupby(['installation_id', 'game_session'])['IsAttemptSuccessful'].transform(np.sum)
    Assess = Assess[['installation_id', 'game_session', 'Attempts', 'Success']].drop_duplicates()
    ratio = Assess['Success'] / Assess['Attempts']
    conditions = [
        (ratio == 1),
        (ratio == 0.5),
        (ratio < 0.5) & (ratio > 0),
        (ratio == 0)]
    choices = [3, 2, 1, 0]
    Assess['accuracy_group'] = np.select(conditions, choices)
    Assess['Past_Assessment_ag'] = round(Assess.groupby('installation_id')['accuracy_group'].shift(1, fill_value=0))
    Assess['Past_Assessment_att'] = round(Assess.groupby('installation_id')['Attempts'].shift(1, fill_value=0))
    Assess['Past_Assessment_succ'] = round(Assess.groupby('installation_id')['Success'].shift(1, fill_value=0))

    Assess['cummean_ag'] = Assess.groupby('installation_id')['Past_Assessment_ag'].transform(lambda x: x.expanding().mean())
    Assess['cummean_att'] = Assess.groupby('installation_id')['Past_Assessment_att'].transform(lambda x: x.expanding().mean())
    Assess['cummean_succ'] = Assess.groupby('installation_id')['Past_Assessment_succ'].transform(lambda x: x.expanding().mean())

    Assess['cumstd_ag'] = Assess.groupby('installation_id')['Past_Assessment_ag'].transform(lambda x: x.expanding().std())
    Assess['cumstd_att'] = Assess.groupby('installation_id')['Past_Assessment_att'].transform(lambda x: x.expanding().std())
    Assess['cumstd_succ'] = Assess.groupby('installation_id')['Past_Assessment_succ'].transform(lambda x: x.expanding().std())

    Assess['cumstd_ag'].fillna(0, inplace=True)
    Assess['cumstd_att'].fillna(0, inplace=True)
    Assess['cumstd_succ'].fillna(0, inplace=True)

    Assess = Assess[
        ['installation_id', 'game_session',
         'Past_Assessment_ag',
         'Past_Assessment_att',
         'Past_Assessment_succ',
         'cummean_ag',
         'cummean_att',
         'cummean_succ',
         'cumstd_ag',
         'cumstd_att',
        'cumstd_succ']]
    # Slice 1
    slice1 = data.copy().loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type', 'title', 'world', 'Total_Game_Session_Time',
                              'Total_Game_Session_Events']].drop_duplicates()
    slice1['Game_Session_Order'] = slice1.groupby('installation_id')['game_session'].cumcount() + 1
    slice1['Cumulative_Time_Spent'] = slice1.groupby(['installation_id'])['Total_Game_Session_Time'].cumsum()
    # Slice 2
    slice2 = data.loc[(data.game_time == data.Total_Game_Session_Time) &
                      (data.event_count == data.Total_Game_Session_Events),
                      ['installation_id', 'game_session', 'type',
                       'Cumulative_Attempts', 'Cumulative_Successes',
                       'Assessments_played_Counter']].copy().drop_duplicates()
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
    slice3['Past_Assessment_Session_Time'] = slice3.groupby('installation_id')['Assessment_Session_Time'].\
        shift(1, fill_value=0)
    slice3['Past_Assessment_NumberOfEvents'] = slice3.groupby('installation_id')['Assessment_NumberOfEvents'].\
        shift(1, fill_value=0)
    slice3 = slice3.assign(cumAverageTime=slice3.groupby('installation_id', sort=False)['Past_Assessment_Session_Time'].\
                           transform(lambda x: x.expanding().mean()))
    slice3 = slice3.assign(cumAverageEvents=slice3.groupby('installation_id', sort=False)['Past_Assessment_NumberOfEvents'].\
                           transform(lambda x: x.expanding().mean()))
    slice3 = slice3.assign(cumsdTime=slice3.groupby('installation_id', sort=False)['Past_Assessment_Session_Time'].\
                           transform(lambda x: x.expanding().std()))
    slice3['cumsdTime'].fillna(0, inplace=True)
    slice3 = slice3.assign(cumsdEvents=slice3.groupby('installation_id', sort=False)['Past_Assessment_NumberOfEvents'].\
                           transform(lambda x: x.expanding().std()))
    slice3['cumsdEvents'].fillna(0, inplace=True)
    slice3 = slice3[
        ['installation_id', 'game_session',
         'Past_Assessment_Session_Time',
         'Past_Assessment_NumberOfEvents',
         'cumAverageTime',
         'cumAverageEvents',
         'cumsdTime',
         'cumsdEvents']]
    # Slice 4
    Cols = ['installation_id', 'game_session', 'type', 'event_count', 'event_code', 'Attempt', 'game_time']
    Event_and_Attempts = data[Cols].copy()
    Event_and_Attempts = Event_and_Attempts[Event_and_Attempts['type'] == 'Assessment']
    Event_and_Attempts['Num_Of_Events_Till_Attempt'] = Event_and_Attempts.loc[Event_and_Attempts.Attempt == 1,
                                                                              'event_count']
    Event_and_Attempts['Num_Of_Events_Till_Attempt'] = Event_and_Attempts['Num_Of_Events_Till_Attempt']. \
        replace(np.nan, 0)
    Event_and_Attempts['Time_past_Till_Attempt'] = Event_and_Attempts.loc[Event_and_Attempts.Attempt == 1,
                                                                          'game_time']
    Event_and_Attempts['Time_past_Till_Attempt'] = Event_and_Attempts['Time_past_Till_Attempt'].replace(np.nan, 0)
    Event_and_Attempts = Event_and_Attempts.groupby(['installation_id', 'game_session'])['Num_Of_Events_Till_Attempt',
                                                                                         'Time_past_Till_Attempt'].max()
    Event_and_Attempts = Event_and_Attempts.reset_index()
    Event_and_Attempts['Prev_Assessment_Time_past_Till_Attempt'] = Event_and_Attempts['Time_past_Till_Attempt']. \
        shift(1, fill_value=0)
    Event_and_Attempts['Prev_Assessment_Num_Of_Events_Till_Attempt'] = Event_and_Attempts['Num_Of_Events_Till_Attempt']. \
        shift(1, fill_value=0)

    Event_and_Attempts = Event_and_Attempts.assign(
        cummean_Assessment_Time_past_Till_Attempt=Event_and_Attempts.groupby('installation_id', sort=False)['Prev_Assessment_Time_past_Till_Attempt'].\
                           transform(lambda x: x.expanding().mean()))
    Event_and_Attempts = Event_and_Attempts.assign(
        cummean_Assessment_Num_Of_Events_Till_Attempt=Event_and_Attempts.groupby('installation_id', sort=False)['Prev_Assessment_Num_Of_Events_Till_Attempt'].\
                           transform(lambda x: x.expanding().mean()))

    Event_and_Attempts = Event_and_Attempts.assign(
        cumsd_Assessment_Time_past_Till_Attempt=Event_and_Attempts.groupby('installation_id', sort=False)['Prev_Assessment_Time_past_Till_Attempt']. \
            transform(lambda x: x.expanding().std()))
    Event_and_Attempts['cumsd_Assessment_Time_past_Till_Attempt'].fillna(0, inplace=True)

    Event_and_Attempts = Event_and_Attempts.assign(
        cumsd_Assessment_Num_Of_Events_Till_Attempt=Event_and_Attempts.groupby('installation_id', sort=False)['Prev_Assessment_Num_Of_Events_Till_Attempt'].\
                           transform(lambda x: x.expanding().std()))
    Event_and_Attempts['cumsd_Assessment_Num_Of_Events_Till_Attempt'].fillna(0, inplace=True)

    Event_and_Attempts = Event_and_Attempts[['installation_id',
                                             'game_session',
                                             'Prev_Assessment_Time_past_Till_Attempt',
                                             'Prev_Assessment_Num_Of_Events_Till_Attempt',
                                             'cummean_Assessment_Num_Of_Events_Till_Attempt',
                                             'cummean_Assessment_Time_past_Till_Attempt',
                                             'cumsd_Assessment_Time_past_Till_Attempt',
                                             'cumsd_Assessment_Num_Of_Events_Till_Attempt']]

    # Slice 5

    title_slice5 = pd.pivot_table(slice1[['installation_id', 'game_session', 'type','title', 'Game_Session_Order']],
                                index=['installation_id', 'game_session','type', 'Game_Session_Order'],
                                columns='title',
                                aggfunc= len,
                                fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    titlecols = slice1.title.unique()
    title_slice5[["visits_" + title for title in titlecols]] = title_slice5.groupby('installation_id')[titlecols].transform(np.nancumsum)
    title_slice5_assessments = title_slice5[title_slice5.type == 'Assessment']
    title_slice5_assessments = title_slice5_assessments.drop(['Game_Session_Order', 'type'], axis = 1)


    # Slice 6


    title_slice6 = pd.pivot_table(slice1[['installation_id', 'game_session', 'type','title', 'Game_Session_Order','Total_Game_Session_Time']],
                                index=['installation_id', 'game_session','type', 'Game_Session_Order'],
                                columns='title',
                                values= 'Total_Game_Session_Time',
                                fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols =["cumulative_timespent_" + title for title in titlecols]
    title_slice6[cols] = title_slice6.groupby('installation_id')[titlecols].transform(np.cumsum)
    title_slice6[cols] = title_slice6[cols].shift(1, fill_value=0)
    cols.extend(['installation_id', 'game_session'])
    title_slice6_assessments = title_slice6.loc[title_slice6.type == 'Assessment', cols]

    # Slice 7

    title_slice7 = pd.pivot_table(slice1[['installation_id', 'game_session', 'type','title', 'Game_Session_Order','Total_Game_Session_Events']],
                                index=['installation_id', 'game_session','type', 'Game_Session_Order'],
                                columns='title',
                                values= 'Total_Game_Session_Events',
                                fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols =["cumulative_events_" + title for title in titlecols]
    title_slice7[cols] = title_slice7.groupby('installation_id')[titlecols].transform(np.cumsum)
    title_slice7[cols] = title_slice7[cols].shift(1, fill_value=0)
    cols.extend(['installation_id', 'game_session'])
    title_slice7_assessments = title_slice7.loc[title_slice7.type == 'Assessment', cols]

    # Slice 8
    slice8 = data.copy().loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type',
                              'title',
                              'Cumulative_Attempts',
                              'Cumulative_Successes'
                              ]].drop_duplicates()

    slice8['Game_Session_Order'] = slice8.groupby('installation_id')['game_session'].cumcount() + 1
    title_slice8 = pd.pivot_table(slice8[['installation_id', 'game_session', 'type','title', 'Game_Session_Order','Cumulative_Attempts']],
                                index=['installation_id', 'game_session','type', 'Game_Session_Order'],
                                columns='title',
                                values= 'Cumulative_Attempts',
                                fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols =["cumulative_attempts_" + title for title in titlecols]
    title_slice8[cols] = title_slice8[titlecols]
    cols.extend(['installation_id', 'game_session'])
    title_slice8_assessments = title_slice8.loc[title_slice8.type == 'Assessment', cols]

    # Slice 9
    title_slice9 = pd.pivot_table(slice8[['installation_id', 'game_session', 'type','title', 'Game_Session_Order','Cumulative_Successes']],
                                index=['installation_id', 'game_session','type', 'Game_Session_Order'],
                                columns='title',
                                values= 'Cumulative_Successes',
                                fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols =["cumulative_successes_" + title for title in titlecols]
    title_slice9[cols] = title_slice9[titlecols]
    cols.extend(['installation_id', 'game_session'])
    title_slice9_assessments = title_slice9.loc[title_slice9.type == 'Assessment', cols]



    # Slice 1 / Type frequency Experience Measures
    type_slice = pd.pivot_table(slice1[['installation_id', 'game_session', 'type', 'Game_Session_Order']],
                                index=['installation_id', 'game_session', 'Game_Session_Order'],
                                columns='type',
                                aggfunc=len,
                                fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    type_slice['Past_Activities'] = type_slice.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice['Past_Games'] = type_slice.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice['Past_Clips'] = type_slice.groupby('installation_id')['Clip'].transform(np.cumsum)
    type_slice['Past_Assessments'] = type_slice.groupby('installation_id')['Assessment'].transform(np.cumsum) - 1
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
        index=['installation_id', 'game_session', 'Game_Session_Order','type'],
        columns='type',
        values='Total_Game_Session_Time',
        aggfunc=sum,
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    type_slice2['Time_spent_on_Activities'] = type_slice2.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice2['Time_spent_on_Games'] = type_slice2.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice2['Time_spent_on_Clips'] = type_slice2.groupby('installation_id')['Clip'].transform(np.cumsum)
    type_slice2['Time_spent_on_Assessments'] = type_slice2.groupby('installation_id')['Assessment'].transform(np.cumsum)



    type_slice2['Average_Time_spent_on_Activities'] = type_slice2.groupby('installation_id')['Activity'].transform(lambda x: x.expanding().mean())
    type_slice2['Average_Time_spent_on_Games'] = type_slice2.groupby('installation_id')['Game'].transform(lambda x: x.expanding().mean())
    type_slice2['Average_Time_spent_on_Clips'] = type_slice2.groupby('installation_id')['Clip'].transform(lambda x: x.expanding().mean())
    type_slice2['Average_Time_spent_on_Assessments'] = type_slice2.groupby('installation_id')['Assessment'].transform(lambda x: x.expanding().mean())

    type_slice2_assessments = type_slice2[type_slice2.type == 'Assessment'].copy()
    type_slice2_assessments.loc[:, 'Total_Time_spent'] = type_slice2_assessments[
        ['Time_spent_on_Activities', 'Time_spent_on_Games', 'Time_spent_on_Clips', 'Time_spent_on_Assessments']].sum(
        axis=1)

    type_slice2_assessments['Average_Time_spent_on_games'] = type_slice2_assessments.groupby('installation_id')['Total_Time_spent'].transform(
        lambda x: x.expanding().mean())

    type_slice2_assessments['Std_Time_spent_on_games'] = type_slice2_assessments.groupby('installation_id')['Total_Time_spent'].transform(
        lambda x: x.expanding().mean())
    type_slice2_assessments = type_slice2_assessments.reset_index()
    type_slice2_assessments = type_slice2_assessments.loc[:,
                              ['installation_id', 'game_session',
                               'Total_Time_spent',
                               'Time_spent_on_Activities',
                               'Time_spent_on_Games',
                               'Time_spent_on_Clips',
                               'Time_spent_on_Assessments',
                               'Average_Time_spent_on_Activities',
                               'Average_Time_spent_on_Games',
                               'Average_Time_spent_on_Clips',
                               'Average_Time_spent_on_Assessments',
                               'Average_Time_spent_on_games',
                               'Std_Time_spent_on_games'
                             ]].drop_duplicates()

    # Slice 1 / world time spent Experience Measures
    world_slice3 = pd.pivot_table(
        slice1[['installation_id', 'game_session', 'world', 'Game_Session_Order', 'type', 'Total_Game_Session_Time']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='world',
        values='Total_Game_Session_Time',
        aggfunc=sum,
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    world_slice3['Time_spent_in_CRYSTALCAVES'] = world_slice3.groupby('installation_id')['CRYSTALCAVES'].transform(
        np.cumsum)
    world_slice3['Time_spent_in_MAGMAPEAK'] = world_slice3.groupby('installation_id')['MAGMAPEAK'].transform(np.cumsum)
    world_slice3['Time_spent_in_TREETOPCITY'] = world_slice3.groupby('installation_id')['TREETOPCITY'].transform(
        np.cumsum)
    world_slice3 = world_slice3[world_slice3.type == 'Assessment']
    world_slice3 = world_slice3[['installation_id', 'game_session', 'Time_spent_in_CRYSTALCAVES',
                                 'Time_spent_in_MAGMAPEAK', 'Time_spent_in_TREETOPCITY']]

    # Substract Level of Player
    Level_slice4 = slice1.copy()
    Level_slice4['Level'] = np.where(Level_slice4['title'].str.contains("Level"),
                                     Level_slice4['title'].str.strip().str[-1], 0)
    Level_slice4['Level'] = pd.to_numeric(Level_slice4['Level'])
    Level_slice4 = pd.pivot_table(
        Level_slice4[['installation_id', 'game_session', 'type', 'world', 'Level', 'Game_Session_Order']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='world',
        values='Level',
        aggfunc=max,
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    Level_slice4['Level_reached_in_CRYSTALCAVES'] = Level_slice4.groupby('installation_id')['CRYSTALCAVES'].transform(
        'cummax')
    Level_slice4['Level_reached_in_MAGMAPEAK'] = Level_slice4.groupby('installation_id')['MAGMAPEAK'].transform(
        'cummax')
    Level_slice4['Level_reached_in_TREETOPCITY'] = Level_slice4.groupby('installation_id')['TREETOPCITY'].transform(
        'cummax')
    Level_slice4 = Level_slice4[Level_slice4.type == 'Assessment']
    Level_slice4 = Level_slice4[['installation_id', 'game_session',
                                 'Level_reached_in_CRYSTALCAVES',
                                 'Level_reached_in_MAGMAPEAK',
                                 'Level_reached_in_TREETOPCITY']]

    # Get all together
    MergedSlices = pd.merge(type_slice_assessments, type_slice2_assessments, on=['installation_id', 'game_session'],
                            how='inner')
    # Create Dummies
    Assessments = data[data.type == 'Assessment'].copy()
    Assessments['timestamp'] = pd.to_datetime(Assessments['timestamp'], format="%Y-%m-%d %H:%M")
    Assessments = Assessments.sort_values('timestamp', ascending=True)
    Assessments = Assessments.drop_duplicates()
    Assessments = af.convert_datetime(Assessments)
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
    FinalData = pd.merge(FinalData, Event_and_Attempts, how='inner',
                         on=['installation_id', 'game_session'])

    FinalData = pd.merge(FinalData, world_slice3, how='inner',
                         on=['installation_id', 'game_session'])
    FinalData = pd.merge(FinalData, Level_slice4, how='inner',
                         on=['installation_id', 'game_session'])
    FinalData = pd.merge(FinalData, Assess, how='inner',
                         on=['installation_id', 'game_session'])
    FinalData = pd.merge(FinalData, title_slice5_assessments, how='inner',
                         on=['installation_id', 'game_session'])
    FinalData = pd.merge(FinalData, title_slice6_assessments, how='inner',
                         on=['installation_id', 'game_session'])
    FinalData = pd.merge(FinalData, title_slice7_assessments, how='inner',
                         on=['installation_id', 'game_session'])
    FinalData = pd.merge(FinalData, title_slice8_assessments, how='inner',
                         on=['installation_id', 'game_session'])
    FinalData = pd.merge(FinalData, title_slice9_assessments, how='inner',
                         on=['installation_id', 'game_session'])
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
    Assess.loc[Assess.order == Assess.LastGame - 1, "To_Predict"] = 1
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
