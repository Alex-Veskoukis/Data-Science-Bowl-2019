# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')


# Any results you write to the current directory are saved as output.


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

    Assess['cummean_ag'] = Assess.groupby('installation_id')['Past_Assessment_ag'].transform(
        lambda x: x.expanding().mean())
    Assess['cummean_att'] = Assess.groupby('installation_id')['Past_Assessment_att'].transform(
        lambda x: x.expanding().mean())
    Assess['cummean_succ'] = Assess.groupby('installation_id')['Past_Assessment_succ'].transform(
        lambda x: x.expanding().mean())

    Assess['cumstd_ag'] = Assess.groupby('installation_id')['Past_Assessment_ag'].transform(
        lambda x: x.expanding().std())
    Assess['cumstd_att'] = Assess.groupby('installation_id')['Past_Assessment_att'].transform(
        lambda x: x.expanding().std())
    Assess['cumstd_succ'] = Assess.groupby('installation_id')['Past_Assessment_succ'].transform(
        lambda x: x.expanding().std())

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
    return Assess


def get_past_attemps_and_successes(data):
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


def get_cummulative_time_spent_in_titles(data):
    import pandas as pd
    titlecols = data.title.unique()
    title_slice6 = pd.pivot_table(
        data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order', 'Total_Game_Session_Time']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='title',
        values='Total_Game_Session_Time',
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols = ["cumulative_timespent_" + title for title in titlecols]
    title_slice6[cols] = title_slice6.groupby('installation_id')[titlecols].transform(np.cumsum)
    title_slice6[cols] = title_slice6[cols].shift(1, fill_value=0)
    cols.extend(['installation_id', 'game_session'])
    title_slice6_assessments = title_slice6.loc[title_slice6.type == 'Assessment', cols]
    return title_slice6_assessments


def get_cummulative_events_seen_per_title(data):
    import pandas as pd
    titlecols = data.title.unique()
    title_slice7 = pd.pivot_table(
        data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order', 'Total_Game_Session_Events']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='title',
        values='Total_Game_Session_Events',
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols = ["cumulative_events_" + title for title in titlecols]
    title_slice7[cols] = title_slice7.groupby('installation_id')[titlecols].transform(np.cumsum)
    title_slice7[cols] = title_slice7[cols].shift(1, fill_value=0)
    cols.extend(['installation_id', 'game_session'])
    title_slice7_assessments = title_slice7.loc[title_slice7.type == 'Assessment', cols]
    return title_slice7_assessments


def get_cummulative_attempts_per_title(data):
    import pandas as pd
    titlecols = data.title.unique()
    title_slice8 = pd.pivot_table(
        data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order', 'Past_Total_Attempts']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='title',
        values='Past_Total_Attempts',
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols = ["cumulative_attempts_" + title for title in titlecols]
    title_slice8[cols] = title_slice8[titlecols]
    cols.extend(['installation_id', 'game_session'])
    title_slice8_assessments = title_slice8.loc[title_slice8.type == 'Assessment', cols]
    return title_slice8_assessments


def get_cummulative_successes_per_title(data):
    import pandas as pd
    titlecols = data.title.unique()
    title_slice9 = pd.pivot_table(
        data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order', 'Past_Total_Successes']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='title',
        values='Past_Total_Successes',
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols = ["cumulative_successes_" + title for title in titlecols]
    title_slice9[cols] = title_slice9[titlecols]
    cols.extend(['installation_id', 'game_session'])
    title_slice9_assessments = title_slice9.loc[title_slice9.type == 'Assessment', cols]
    return title_slice9_assessments


def get_frequency_per_type(data):
    import pandas as pd
    type_slice = pd.pivot_table(data[['installation_id', 'game_session', 'type', 'Game_Session_Order']],
                                index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
                                columns='type',
                                aggfunc=len,
                                fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    type_slice['Activities_played'] = type_slice.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice['Games_played'] = type_slice.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice['Clips__played'] = type_slice.groupby('installation_id')['Clip'].transform(np.cumsum)
    type_slice['Assessments_played'] = type_slice.groupby('installation_id')['Assessment'].transform(np.cumsum)
    type_slice_assessments = type_slice[type_slice.Assessment == 1]
    type_slice_assessments = type_slice_assessments.rename(columns={'Game_Session_Order': 'Total_Games_played'})
    type_slice_assessments = type_slice_assessments.drop(['Game', 'Clip', 'Assessment', 'Activity'], axis=1)
    type_slice_assessments = type_slice_assessments.loc[:,
                             ['installation_id', 'game_session', 'Total_Games_played', 'Clips__played',
                              'Games_played', 'Assessments_played', 'Activities_played']].drop_duplicates()
    return type_slice_assessments


def get_cumulative_time_spent_on_types(data):
    import pandas as pd
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
    import pandas as pd
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
    import pandas as pd
    Assessments = data[data.type == 'Assessment'].copy()
    Assessments['timestamp'] = pd.to_datetime(Assessments['timestamp'], format="%Y-%m-%d %H:%M")
    Assessments = Assessments.sort_values('timestamp', ascending=True)
    Assessments = Assessments.drop_duplicates()
    Assessments = convert_datetime(Assessments)
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
    return Assessments


def get_vists_per_title(data):
    import pandas as pd
    title_slice5 = pd.pivot_table(data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order']],
                                  index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
                                  columns='title',
                                  aggfunc=len,
                                  fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    titlecols = data.title.unique()
    title_slice5[["visits_" + title for title in titlecols]] = \
        title_slice5.groupby('installation_id')[titlecols].transform(np.nancumsum)
    title_slice5_assessments = title_slice5[title_slice5.type == 'Assessment']

    cols = ["visits_" + title for title in titlecols]
    cols.extend(['installation_id', 'game_session'])
    titles_played = title_slice5_assessments[cols]
    return titles_played



def get_cummulative_time_spent_in_titles(data):
    import pandas as pd
    titlecols = data.title.unique()
    title_slice6 = pd.pivot_table(
        data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order', 'Total_Game_Session_Time']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='title',
        values='Total_Game_Session_Time',
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols = ["cumulative_timespent_" + title for title in titlecols]
    title_slice6[cols] = title_slice6.groupby('installation_id')[titlecols].transform(np.cumsum)
    title_slice6[cols] = title_slice6[cols].shift(1, fill_value=0)
    cols.extend(['installation_id', 'game_session'])
    title_slice6_assessments = title_slice6.loc[title_slice6.type == 'Assessment', cols]
    return title_slice6_assessments


def get_cummulative_events_seen_per_title(data):
    import pandas as pd
    titlecols = data.title.unique()
    title_slice7 = pd.pivot_table(
        data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order', 'Total_Game_Session_Events']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='title',
        values='Total_Game_Session_Events',
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols = ["cumulative_events_" + title for title in titlecols]
    title_slice7[cols] = title_slice7.groupby('installation_id')[titlecols].transform(np.cumsum)
    title_slice7[cols] = title_slice7[cols].shift(1, fill_value=0)
    cols.extend(['installation_id', 'game_session'])
    title_slice7_assessments = title_slice7.loc[title_slice7.type == 'Assessment', cols]
    return title_slice7_assessments


def get_cummulative_attempts_per_title(data):
    import pandas as pd
    titlecols = data.title.unique()
    title_slice8 = pd.pivot_table(
        data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order', 'Past_Total_Attempts']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='title',
        values='Past_Total_Attempts',
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols = ["cumulative_attempts_" + title for title in titlecols]
    title_slice8[cols] = title_slice8[titlecols]
    cols.extend(['installation_id', 'game_session'])
    title_slice8_assessments = title_slice8.loc[title_slice8.type == 'Assessment', cols]
    return title_slice8_assessments


def get_cummulative_successes_per_title(data):
    import pandas as pd
    titlecols = data.title.unique()
    title_slice9 = pd.pivot_table(
        data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order', 'Past_Total_Successes']],
        index=['installation_id', 'game_session', 'type', 'Game_Session_Order'],
        columns='title',
        values='Past_Total_Successes',
        fill_value=0).reset_index().sort_values(['installation_id', 'Game_Session_Order'])
    cols = ["cumulative_successes_" + title for title in titlecols]
    title_slice9[cols] = title_slice9[titlecols]
    cols.extend(['installation_id', 'game_session'])
    title_slice9_assessments = title_slice9.loc[title_slice9.type == 'Assessment', cols]
    return title_slice9_assessments



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

    data['Assessment_Session_Time'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])[
        'game_time'].transform(np.max)
    data['Assessment_NumberOfEvents'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])[
        'event_count'].transform(np.max)
    # Previous Accuracy
    previous_accuracy_metrics = get_previous_ac_metrics(data)
    print('previous_accuracy_metrics')
    # Slice 1
    slice1 = data.copy().loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type', 'title', 'world', 'Total_Game_Session_Time',
                              'Total_Game_Session_Events']].drop_duplicates()
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

    title_visits = get_vists_per_title(slice1)
    print('title_visits')
    cummulative_time_spent_in_titles = get_cummulative_time_spent_in_titles(slice1)
    print('cummulative_time_spent_in_titles')
    # Slice 1 / events count
    cummulative_events_seen_per_title = get_cummulative_events_seen_per_title(slice1)
    print('cummulative_events_seen_per_title')

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
        slice8[slice8.type == 'Assessment'].groupby('installation_id')['Assessments_played_Counter'].shift(1,
                                                                                                           fill_value=0))

    slice8['Game_Session_Order'] = slice8.groupby('installation_id')['game_session'].cumcount() + 1

    cummulative_attempts_per_title = get_cummulative_attempts_per_title(slice8)
    print('cummulative_attempts_per_title')
    cummulative_successes_per_title = get_cummulative_successes_per_title(slice8)
    print('cummulative_successes_per_title')
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
            title_visits,
            cummulative_time_spent_in_titles,
            cummulative_events_seen_per_title,
            cummulative_attempts_per_title,
            cummulative_successes_per_title]
    FinalData = reduce(lambda left, right: pd.merge(left, right, how='inner', on=['installation_id', 'game_session']),
                       Sets)
    return FinalData


Final = create_features(train)

FinalTrain = pd.merge(Final,
                      train_labels[['installation_id', 'game_session', 'accuracy_group']],
                      how='inner',
                      on=['installation_id', 'game_session'])

FinalTrain = FinalTrain.set_index(['installation_id', 'game_session'])

Test_Features = create_features(test)


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


Test_Set = get_last_assessment(test)

Test_set_full = pd.merge(Test_Features, Test_Set, on=['installation_id', 'game_session'], how='right')

X_train = FinalTrain.loc[:, ~FinalTrain.columns.isin(['accuracy_group', 'installation_id', 'game_session'])]
Y_train = FinalTrain['accuracy_group'].astype(int)

X_test = Test_set_full.set_index(['installation_id', 'game_session'])

model= RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
model.fit(X_train, Y_train)
Y_pred_test = model.predict(X_test)


submission = pd.DataFrame({"installation_id": X_test.reset_index(1).index.values,
                           "accuracy_group": Y_pred_test})
submission.to_csv("submission.csv", index=False)