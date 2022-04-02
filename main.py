# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
import json
import statistics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')


# Any results you write to the current directory are saved as output.
def extract_mean_duration(data):
    df = data[['installation_id','game_session','type','event_data','Assessments_played_Counter']].copy()
    df = df[df.type == 'Assessment']
    df['duration'] = df['event_data'].apply(lambda x: json.loads(x).get('duration'))
    df['mean_duration'] = df.groupby(['installation_id','game_session'])['duration'].transform('mean')
    dt1 = df[['installation_id','game_session','Assessments_played_Counter','mean_duration']].drop_duplicates()
    dt1 = dt1.sort_values(['installation_id','Assessments_played_Counter'], ascending=[True, True])
    dt1['mean_duration'] = dt1['mean_duration'].shift(1, fill_value=0)
    dt1 = dt1.fillna(0)
    x = dt1[['installation_id','game_session','mean_duration']]
    return x

def convert_datetime(df):
    import pandas as pd
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df


def get_previous_ac_metrics(data):
    Assess = data.sort_values(['installation_id', 'timestamp', 'game_session', 'Assessments_played_Counter'],
                              ascending=[True, True, True, True]).copy()
    Assess = Assess[Assess.type == 'Assessment']
    Assess['Attempts'] = Assess.groupby(['installation_id', 'game_session'])['Attempt'].transform(np.sum)
    Assess['Success'] = Assess.groupby(['installation_id', 'game_session'])['IsAttemptSuccessful'].transform(np.sum)
    Assess['Fails'] = Assess['Attempts'] - Assess['Success']
    Assess = Assess[['installation_id',
                     'game_session',
                     'Assessments_played_Counter',
                     'Attempts',
                     'Success',
                     'Fails']].drop_duplicates()
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
    Assess['Past_Assessment_fails'] = round(Assess.groupby('installation_id')['Fails'].shift(1, fill_value=0))
    Assess['Past_Assessment_accuracy'] = Assess['Past_Assessment_succ'] / Assess['Past_Assessment_att']
    Assess['Past_Assessment_accuracy'].fillna(0, inplace=True)
    Assess['cummean_ag'] = Assess.groupby('installation_id')['Past_Assessment_ag'].transform(
        lambda x: x.expanding().mean())
    Assess['cummean_att'] = Assess.groupby('installation_id')['Past_Assessment_att'].transform(
        lambda x: x.expanding().mean())
    Assess['cummean_succ'] = Assess.groupby('installation_id')['Past_Assessment_succ'].transform(
        lambda x: x.expanding().mean())
    Assess['cummean_fails'] = Assess.groupby('installation_id')['Past_Assessment_fails'].transform(
        lambda x: x.expanding().mean())
    Assess['cummean_accuracy'] = Assess.groupby('installation_id')['Past_Assessment_accuracy'].transform(
        lambda x: x.expanding().mean())

    Assess['cumstd_ag'] = Assess.groupby('installation_id')['Past_Assessment_ag'].transform(
        lambda x: x.expanding().std())
    Assess['cumstd_att'] = Assess.groupby('installation_id')['Past_Assessment_att'].transform(
        lambda x: x.expanding().std())
    Assess['cumstd_succ'] = Assess.groupby('installation_id')['Past_Assessment_succ'].transform(
        lambda x: x.expanding().std())
    Assess['cumstd_fails'] = Assess.groupby('installation_id')['Past_Assessment_fails'].transform(
        lambda x: x.expanding().std())
    Assess['cumstd_accuracy'] = Assess.groupby('installation_id')['Past_Assessment_accuracy'].transform(
        lambda x: x.expanding().std())

    Assess['cumstd_ag'].fillna(0, inplace=True)
    Assess['cumstd_att'].fillna(0, inplace=True)
    Assess['cumstd_succ'].fillna(0, inplace=True)
    Assess['cumstd_fails'].fillna(0, inplace=True)
    Assess['cumstd_accuracy'].fillna(0, inplace=True)

    pastAG = Assess[['installation_id', 'game_session', 'Assessments_played_Counter', 'Past_Assessment_ag']]
    Assess2 = pd.pivot_table(pastAG,
                             index=['installation_id', 'game_session', 'Assessments_played_Counter'],
                             columns='Past_Assessment_ag',
                             values='Past_Assessment_ag',
                             aggfunc='nunique').reset_index()
    Assess2 = Assess2.rename(columns={0: "zeros", 1: "ones", 2: "twos", 3: 'threes'})
    Assess2 = Assess2.sort_values(['installation_id', 'Assessments_played_Counter'])
    Assess2.loc[:, ["zeros", 'ones', "twos", 'threes']] = Assess2.groupby('installation_id')[
        "zeros", 'ones', "twos", 'threes'].apply(np.cumsum)
    Assess2.loc[:, ["zeros", 'ones', "twos", 'threes']] = Assess2.groupby('installation_id')[
        "zeros", 'ones', "twos", 'threes'].fillna(method='ffill')
    Assess2.loc[:, ["zeros", 'ones', "twos", 'threes']] = Assess2.loc[:, ["zeros", 'ones', "twos", 'threes']].fillna(0)
    Assess2 = Assess2[['installation_id', 'game_session', "zeros", 'ones', "twos", 'threes']]
    Assess = Assess[
        ['installation_id', 'game_session',
         'Past_Assessment_ag',
         'Past_Assessment_att',
         'Past_Assessment_succ',
         'Past_Assessment_fails',
         'cumstd_fails',
         'cummean_fails',
         'Past_Assessment_accuracy',
         'cummean_accuracy',
         'cumstd_accuracy',
         'cummean_ag',
         'cummean_att',
         'cummean_succ',
         'cumstd_ag',
         'cumstd_att',
         'cumstd_succ']]
    AssessFinal = pd.merge(Assess, Assess2, on=['installation_id', 'game_session'], how='inner')
    return AssessFinal


def get_past_attemps_and_successes(data):
    slice2 = data.loc[(data.game_time == data.Total_Game_Session_Time) &
                      (data.event_count == data.Total_Game_Session_Events),
                      ['installation_id', 'game_session', 'type',
                       'Cumulative_Attempts', 'Cumulative_Successes', 'Cumulative_Fails',
                       'Assessments_played_Counter']].copy().drop_duplicates()
    slice2['Game_Session_Order'] = slice2.groupby('installation_id')['game_session'].cumcount() + 1
    slice2 = slice2.sort_values(['installation_id', 'Game_Session_Order'])
    slice2 = slice2[slice2.type == 'Assessment']
    slice2['Past_Total_Attempts'] = round(
        slice2.groupby('installation_id')['Cumulative_Attempts'].shift(1, fill_value=0)).astype('int')
    slice2['Past_Total_Successes'] = round(
        slice2.groupby('installation_id')['Cumulative_Successes'].shift(1, fill_value=0)).astype('int')
    slice2['Past_Total_Fails'] = round(
        slice2.groupby('installation_id')['Cumulative_Fails'].shift(1, fill_value=0)).astype('int')
    slice2['Past_Assessments_Played'] = round(
        slice2.groupby('installation_id')['Assessments_played_Counter'].shift(1, fill_value=0)).astype('int')

    slice2['Past_Total_Accuracy'] = slice2['Past_Total_Successes'] / slice2['Past_Total_Attempts']
    slice2['Past_Total_Accuracy'].fillna(0, inplace=True)
    slice2 = slice2.loc[:, ['installation_id', 'game_session',
                            'Past_Total_Attempts', 'Past_Total_Successes', 'Past_Total_Accuracy',
                            'Past_Total_Fails', 'Past_Assessments_Played']]
    return slice2


def get_past_assessment_time_events_and_metrics(data):
    slice3 = data.loc[data.type == 'Assessment', ['installation_id',
                                                  'game_session',
                                                  'Assessment_Session_Time',
                                                  'Assessment_NumberOfEvents']].drop_duplicates()
    slice3['Past_Assessment_Session_Time'] = slice3.groupby('installation_id')['Assessment_Session_Time']. \
        shift(1, fill_value=0)
    slice3['Past_Assessment_NumberOfEvents'] = slice3.groupby('installation_id')['Assessment_NumberOfEvents']. \
        shift(1, fill_value=0)
    slice3 = slice3.assign(
        cumAverageTime=slice3.groupby('installation_id', sort=False)['Past_Assessment_Session_Time']. \
            transform(lambda x: x.expanding().mean()))
    slice3 = slice3.assign(
        cumAverageEvents=slice3.groupby('installation_id', sort=False)['Past_Assessment_NumberOfEvents']. \
            transform(lambda x: x.expanding().mean()))
    slice3 = slice3.assign(cumsdTime=slice3.groupby('installation_id', sort=False)['Past_Assessment_Session_Time']. \
                           transform(lambda x: x.expanding().std()))
    slice3['cumsdTime'].fillna(0, inplace=True)
    slice3 = slice3.assign(cumsdEvents=slice3.groupby('installation_id', sort=False)['Past_Assessment_NumberOfEvents']. \
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
    return (slice3)


def get_prev_events_and_time_till_attempt(data):
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
        cummean_Assessment_Time_past_Till_Attempt=
        Event_and_Attempts.groupby('installation_id', sort=False)['Prev_Assessment_Time_past_Till_Attempt']. \
            transform(lambda x: x.expanding().mean()))
    Event_and_Attempts = Event_and_Attempts.assign(
        cummean_Assessment_Num_Of_Events_Till_Attempt=
        Event_and_Attempts.groupby('installation_id', sort=False)['Prev_Assessment_Num_Of_Events_Till_Attempt']. \
            transform(lambda x: x.expanding().mean()))

    Event_and_Attempts = Event_and_Attempts.assign(
        cumsd_Assessment_Time_past_Till_Attempt=
        Event_and_Attempts.groupby('installation_id', sort=False)['Prev_Assessment_Time_past_Till_Attempt']. \
            transform(lambda x: x.expanding().std()))
    Event_and_Attempts['cumsd_Assessment_Time_past_Till_Attempt'].fillna(0, inplace=True)

    Event_and_Attempts = Event_and_Attempts.assign(
        cumsd_Assessment_Num_Of_Events_Till_Attempt=
        Event_and_Attempts.groupby('installation_id', sort=False)['Prev_Assessment_Num_Of_Events_Till_Attempt']. \
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
    return Event_and_Attempts


def get_frequency_per_type(data):
    type_slice = pd.pivot_table(data[['installation_id', 'game_session', 'type', 'Game_Session_Order']],
                                index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
                                columns='type',
                                aggfunc=len,
                                fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    type_slice['Activities_played'] = type_slice.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice['Games_played'] = type_slice.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice['Clips__played'] = type_slice.groupby('installation_id')['Clip'].transform(np.cumsum)
    type_slice_assessments = type_slice[type_slice.Assessment == 1]
    type_slice_assessments = type_slice_assessments.rename(columns={'Game_Session_Order': 'Total_Games_played'})
    type_slice_assessments = type_slice_assessments.drop(['Game', 'Clip', 'Assessment', 'Activity'], axis=1)
    type_slice_assessments = type_slice_assessments.loc[:,
                             ['installation_id', 'game_session', 'Total_Games_played', 'Clips__played',
                              'Games_played', 'Activities_played']].drop_duplicates()
    return type_slice_assessments


def get_cumulative_time_spent_on_types(data):
    type_slice2 = pd.pivot_table(
        data[['installation_id', 'game_session', 'type', 'Game_Session_Order', 'Total_Game_Session_Time']],
        index=['installation_id', 'game_session', 'Game_Session_Order', 'type'],
        columns='type',
        values='Total_Game_Session_Time',
        aggfunc=sum,
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    type_slice2['Time_spent_on_Activities'] = type_slice2.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice2['Time_spent_on_Games'] = type_slice2.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice2['Time_spent_on_Assessments'] = type_slice2.groupby('installation_id')['Assessment'].transform(np.cumsum)

    type_slice2['Average_Time_spent_on_Activities'] = \
        type_slice2[type_slice2.type == 'Activity'].groupby('installation_id')['Activity'].transform(
            lambda x: x.expanding().mean())
    type_slice2['Average_Time_spent_on_Activities'] = type_slice2.groupby('installation_id')[
        'Average_Time_spent_on_Activities'].fillna(method='ffill')
    type_slice2['Average_Time_spent_on_Activities'] = type_slice2['Average_Time_spent_on_Activities'].fillna(0)
    type_slice2['Average_Time_spent_on_Activities'] = type_slice2.groupby('installation_id')[
        'Average_Time_spent_on_Activities'].shift(1, fill_value=0)

    type_slice2['Average_Time_spent_on_Games'] = \
        type_slice2[type_slice2.type == 'Game'].groupby('installation_id')['Game'].transform(
            lambda x: x.expanding().mean())
    type_slice2['Average_Time_spent_on_Games'] = type_slice2.groupby('installation_id')[
        'Average_Time_spent_on_Games'].fillna(method='ffill')
    type_slice2['Average_Time_spent_on_Games'] = type_slice2['Average_Time_spent_on_Games'].fillna(0)
    type_slice2['Average_Time_spent_on_Games'] = type_slice2.groupby('installation_id')[
        'Average_Time_spent_on_Games'].shift(1, fill_value=0)

    type_slice2['Average_Time_spent_on_Assessments'] = \
        type_slice2[type_slice2.type == 'Assessment'].groupby('installation_id')['Assessment'].transform(
            lambda x: x.expanding().mean())
    type_slice2['Average_Time_spent_on_Assessments'] = type_slice2.groupby('installation_id')[
        'Average_Time_spent_on_Assessments'].fillna(method='ffill')
    type_slice2['Average_Time_spent_on_Assessments'] = type_slice2['Average_Time_spent_on_Assessments'].fillna(0)
    type_slice2['Average_Time_spent_on_Assessments'] = type_slice2.groupby('installation_id')[
        'Average_Time_spent_on_Assessments'].shift(1, fill_value=0)

    type_slice2_assessments = type_slice2[type_slice2.type == 'Assessment'].copy()
    type_slice2_assessments.loc[:, 'Total_Time_spent'] = type_slice2_assessments[
        ['Time_spent_on_Activities', 'Time_spent_on_Games', 'Time_spent_on_Assessments']].sum(axis=1)

    type_slice2_assessments['Average_Time_spent_on_games'] = \
        type_slice2_assessments.groupby('installation_id')['Total_Time_spent'].transform(lambda x: x.expanding().mean())

    type_slice2_assessments['Std_Time_spent_on_games'] = \
        type_slice2_assessments.groupby('installation_id')['Total_Time_spent'].transform(lambda x: x.expanding().std())
    type_slice2_assessments = type_slice2_assessments.reset_index()
    type_slice2_assessments['Std_Time_spent_on_games'] = type_slice2_assessments['Std_Time_spent_on_games'].fillna(0)
    type_slice2_assessments = type_slice2_assessments.loc[:,
                              ['installation_id', 'game_session',
                               'Total_Time_spent',
                               'Time_spent_on_Activities',
                               'Time_spent_on_Games',
                               'Time_spent_on_Assessments',
                               'Average_Time_spent_on_Activities',
                               'Average_Time_spent_on_Games',
                               'Average_Time_spent_on_Assessments',
                               'Average_Time_spent_on_games',
                               'Std_Time_spent_on_games'
                               ]].drop_duplicates()
    return type_slice2_assessments


def get_time_spent_on_diffrent_worlds(data):
    world_slice3 = pd.pivot_table(
        data[['installation_id', 'game_session', 'world', 'Game_Session_Order', 'type', 'Total_Game_Session_Time']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='world',
        values='Total_Game_Session_Time',
        aggfunc=sum,
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    world_slice3['Time_spent_in_CRYSTALCAVES'] = world_slice3.groupby('installation_id')['CRYSTALCAVES'].transform(
        np.cumsum)
    world_slice3['Time_spent_in_MAGMAPEAK'] = world_slice3.groupby('installation_id')[
        'MAGMAPEAK'].transform(np.cumsum)
    world_slice3['Time_spent_in_TREETOPCITY'] = world_slice3.groupby('installation_id')['TREETOPCITY'].transform(
        np.cumsum)
    world_slice3 = world_slice3[world_slice3.type == 'Assessment']
    world_slice3 = world_slice3[['installation_id', 'game_session', 'Time_spent_in_CRYSTALCAVES',
                                 'Time_spent_in_MAGMAPEAK', 'Time_spent_in_TREETOPCITY']]
    return world_slice3


def substract_level(data):
    import pandas as pd
    Level_slice4 = data.copy()
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
    Level_slice4['Total_Level'] = Level_slice4['Level_reached_in_CRYSTALCAVES'] + \
                                  Level_slice4['Level_reached_in_MAGMAPEAK'] + \
                                  Level_slice4['Level_reached_in_TREETOPCITY']
    Level_slice4 = Level_slice4[['installation_id', 'game_session',
                                 'Level_reached_in_CRYSTALCAVES',
                                 'Level_reached_in_MAGMAPEAK',
                                 'Level_reached_in_TREETOPCITY',
                                 'Total_Level']]
    return Level_slice4


def create_world_time_assesstitle_Dummies(data):
    Assessments = data[data.type == 'Assessment'].copy()
    Assessments['timestamp'] = pd.to_datetime(Assessments['timestamp'], format="%Y-%m-%d %H:%M")
    Assessments = Assessments.sort_values('timestamp', ascending=True)
    Assessments = Assessments.drop_duplicates()
    Assessments = convert_datetime(Assessments)
    del Assessments['timestamp']
    Assessments['title'] = Assessments['title'].str.rstrip(' (Assessment)')
    Assessments = Assessments.set_index(['installation_id', 'game_session'])
    Assessments = Assessments[['title', 'world', 'month', 'hour', 'dayofweek']]
    Assessments['title'] = pd.Categorical(Assessments['title'])
    Assessments['world'] = pd.Categorical(Assessments['world'])
    Assessments = pd.concat([Assessments, pd.get_dummies(Assessments['world'])], axis=1)
    del Assessments['world']
    Assessments = pd.concat([Assessments, pd.get_dummies(Assessments['title'])], axis=1)
    del Assessments['title']
    Assessments = Assessments.reset_index()
    Assessments = Assessments.drop_duplicates()
    Assessments = Assessments[
        Assessments.hour == Assessments.groupby(['installation_id', 'game_session'])['hour'].transform('min')]
    return Assessments


def get_last_assessment(data):
    Assess = data[data.type == 'Assessment'].copy()
    Assess = Assess[['installation_id', 'game_session']]
    Assess["To_Predict"] = 0
    Assess['order'] = Assess.groupby('installation_id')[
        'game_session'].transform(lambda x: np.round(pd.factorize(x)[0] + 1))
    Assess['LastGame'] = Assess.groupby('installation_id')['order'].transform('max')
    Assess.loc[Assess.order == Assess.LastGame, "To_Predict"] = 1
    Assess = Assess.drop_duplicates()
    Assess = Assess.loc[Assess.To_Predict == 1, ['installation_id', 'game_session']]
    return Assess


def get_final_train(dt, type = 'down'):
    FinalTrain = dt.copy()
    if type == 'down':
        count = min(FinalTrain.accuracy_group.value_counts())
    else:
        count = max(FinalTrain.accuracy_group.value_counts())

    # Divide by class
    df_class_0 = FinalTrain[FinalTrain['accuracy_group'] == 0]
    df_class_1 = FinalTrain[FinalTrain['accuracy_group'] == 1]
    df_class_2 = FinalTrain[FinalTrain['accuracy_group'] == 2]
    df_class_3 = FinalTrain[FinalTrain['accuracy_group'] == 3]

    df_class_0_under = df_class_0.sample(count, replace=True)
    df_class_1_under = df_class_1.sample(count, replace=True)
    df_class_2_under = df_class_2.sample(count, replace=True)
    df_class_3_under = df_class_3.sample(count, replace=True)

    FinalTrain = pd.concat([df_class_0_under, df_class_1_under, df_class_2_under, df_class_3_under], axis=0)
    return FinalTrain



def leastFrequent(arr):
    arr = np.asarray(arr)
    n = len(arr)
    # Sort the array
    arr.sort()

    # find the min frequency using
    # linear traversal
    min_count = n + 1
    res = -1
    curr_count = 1
    for i in range(1, n):
        if (arr[i] == arr[i - 1]):
            curr_count = curr_count + 1
        else:
            if (curr_count < min_count):
                min_count = curr_count
                res = arr[i - 1]

            curr_count = 1

    # If last element is least frequent
    if (curr_count < min_count):
        min_count = curr_count
        res = arr[n - 1]

    return res

def overall_measures(data):
    dt = data[[ 'type','title', 'Total_Game_Session_Time','Attempt','IsAttemptSuccessful']].drop_duplicates()
    dt = dt.sort_values([ 'type', 'title'])
    dt = dt[dt.type == 'Assessment']
    dt1 = dt.groupby('title')['Total_Game_Session_Time'].mean().reset_index(name='OverallMean_Game_Session_Time')
    dt2 = dt.groupby('title')['Attempt'].mean().reset_index(name='OverallMean_Attempt')
    dt3 = dt.groupby('title')['IsAttemptSuccessful'].mean().reset_index(name='OverallMean_IsAttemptSuccessful')
    dt3['OverallMean_accuracy'] = dt3['OverallMean_IsAttemptSuccessful'] / dt2['OverallMean_Attempt']
    Final_dt = reduce(lambda left, right: pd.merge(left, right, how='inner', on=['title']), [dt1,dt2,dt3])
    return Final_dt


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
    # initial measures
    data['Total_Game_Session_Time'] = Inst_Game_Group['game_time'].transform(np.max)
    data['Total_Game_Session_Events'] = Inst_Game_Group['event_count'].transform(np.max)
    data['Assessments_played_Counter'] = data[data.type == 'Assessment'].groupby('installation_id')[
        'game_session'].transform(lambda x: np.round(pd.factorize(x)[0] + 1))
    data['Cumulative_Attempts'] = Inst_Group['Attempt'].transform(np.cumsum)
    data['Cumulative_Successes'] = Inst_Group['IsAttemptSuccessful'].transform(np.nancumsum)
    data['Cumulative_Successes'] = Inst_Group['IsAttemptSuccessful'].transform(np.nancumsum)
    data['Cumulative_Fails'] = data['Cumulative_Attempts'] - data['Cumulative_Successes']
    data['Assessment_Session_Time'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])[
        'game_time'].transform(np.max)
    data['Assessment_NumberOfEvents'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])[
        'event_count'].transform(np.max)

    overall = pd.merge(data.loc[data.type == 'Assessment',['installation_id', 'game_session',  'title']].drop_duplicates(),
                        overall_measures(data),
                        on= 'title',
                        how='inner')
    del overall['title']
    print('overall')
    mean_duration = extract_mean_duration(data)
    print('mean_duration')
    # Previous Accuracy
    previous_accuracy_metrics = get_previous_ac_metrics(data)
    print('previous_accuracy_metrics')
    # Slice 1
    slice1 = data.loc[(data.game_time == data.Total_Game_Session_Time) &
                      (data.event_count == data.Total_Game_Session_Events),
                      ['installation_id', 'game_session', 'type', 'title', 'world', 'Total_Game_Session_Time',
                       'Total_Game_Session_Events']].drop_duplicates().copy()
    slice1['Game_Session_Order'] = slice1.groupby('installation_id')['game_session'].cumcount() + 1
    slice1['Cumulative_Time_Spent'] = slice1.groupby(['installation_id'])['Total_Game_Session_Time'].cumsum()
    # Slice 2
    Number_of_attemps_and_successes = get_past_attemps_and_successes(data)
    print('Number_of_attemps_and_successes')
    # Slice 3
    past_assessment_time_events_and_metrics = get_past_assessment_time_events_and_metrics(data)
    print('past_assessment_time_events_and_metrics')
    # Event_and_Attempts
    pre_time_till_attempt_metrics = get_prev_events_and_time_till_attempt(data)
    print('pre_time_till_attempt_metrics')
    Number_of_games_played_per_type = get_frequency_per_type(slice1)
    print('Number_of_games_played_per_type')
    Time_spent_on_games_metrics = get_cumulative_time_spent_on_types(slice1)
    print('Time_spent_on_games_metrics')
    time_spent_on_diffrent_worlds = get_time_spent_on_diffrent_worlds(slice1)
    print('time_spent_on_diffrent_worlds')
    Level_reached = substract_level(slice1)
    print('Level_reached')
    world_time_gametitles_dummies = create_world_time_assesstitle_Dummies(data)
    print('world_time_gametitles_dummies')
    Sets = [Number_of_games_played_per_type,
            Time_spent_on_games_metrics,
            world_time_gametitles_dummies,
            Number_of_attemps_and_successes,
            past_assessment_time_events_and_metrics,
            pre_time_till_attempt_metrics,
            time_spent_on_diffrent_worlds,
            Level_reached,
            previous_accuracy_metrics,
            overall,
            mean_duration]
    FinalData = reduce(lambda left, right: pd.merge(left, right, how='inner', on=['installation_id', 'game_session']),
                       Sets)
    return FinalData


Final = create_features(train)

FinalTrain = pd.merge(Final,
                      train_labels[['installation_id', 'game_session', 'accuracy_group']],
                      how='inner',
                      on=['installation_id', 'game_session'])

FinalTrain = FinalTrain.set_index(['installation_id', 'game_session'])

cor = FinalTrain.corr()

cor_target = abs(cor["accuracy_group"])

relevant_features = cor_target[cor_target>0.05]
relevant_features = relevant_features.index
Test_Features = create_features(test)

Test_Set = get_last_assessment(test)

Test_set_full = pd.merge(Test_Features, Test_Set.loc[:, ~Test_Set.columns.isin(['accuracy'])],
                         on=['installation_id', 'game_session'], how='right')


X_test = Test_set_full.set_index(['installation_id', 'game_session'])
X_test = X_test.loc[:, X_test.columns.isin(relevant_features)]

predictions = pd.DataFrame()
for i in range(0, 40):
    print(i)
    FinalTrain = get_final_train(FinalTrain, type = 'over')
    X_train = FinalTrain.loc[:, ~FinalTrain.columns.isin(['accuracy_group', 'installation_id', 'game_session'])]
    X_train = X_train.loc[:, X_train.columns.isin(relevant_features)]
    Y_train = FinalTrain['accuracy_group'].astype(int)
    model = RandomForestClassifier(n_estimators=10000, n_jobs=-1, random_state=42, max_depth=6)
    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)
    new_pred = pd.DataFrame(data=Y_pred_test, columns=['pred_' + str(i)])
    predictions = pd.concat([predictions, new_pred], axis=1)

predictions['pred'] = predictions.mode(axis=1,numeric_only=True)[0]

Y_pred_test  = predictions['pred'].astype(int)

submission = pd.DataFrame({"installation_id": X_test.reset_index(1).index.values,
                           "accuracy_group": Y_pred_test})

submission[['installation_id', 'accuracy_group']].to_csv("submission.csv", index=False)
