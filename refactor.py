# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from functools import reduce
from sklearn.ensemble import RandomForestClassifier
import auxiliary_functions as af
import auxiliary_functions as af
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')
train_labels = pd.read_csv('Data/train_labels.csv')
# train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
# test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
# train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

# Any results you write to the current directory are saved as output.


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
    previous_accuracy_metrics = af.get_previous_ac_metrics(data)
    print('previous_accuracy_metrics')
    # Slice 1
    slice1 = data.copy().loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type', 'title', 'world', 'Total_Game_Session_Time',
                              'Total_Game_Session_Events']].drop_duplicates()
    slice1['Game_Session_Order'] = slice1.groupby('installation_id')['game_session'].cumcount() + 1
    slice1['Cumulative_Time_Spent'] = slice1.groupby(['installation_id'])['Total_Game_Session_Time'].cumsum()
    # Slice 2
    Number_of_attemps_and_successes = af.get_past_attemps_and_successes(data)
    print('Number_of_attemps_and_successes')
    # Slice 3
    past_assessment_time_events_and_metrics = af.get_past_assessment_time_events_and_metrics(data)
    print('past_assessment_time_events_and_metrics')
    # Event_and_Attempts
    pre_time_till_attempt_metrics = af.get_prev_events_and_time_till_attempt(data)
    print('pre_time_till_attempt_metrics')

    title_visits = af.get_vists_per_title(slice1)
    print('title_visits')
    cummulative_time_spent_in_titles = af.get_cummulative_time_spent_in_titles(slice1)
    print('cummulative_time_spent_in_titles')
    # Slice 1 / events count
    cummulative_events_seen_per_title = af.get_cummulative_events_seen_per_title(slice1)
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

    cummulative_attempts_per_title = af.get_cummulative_attempts_per_title(slice8)
    print('cummulative_attempts_per_title')
    cummulative_successes_per_title = af.get_cummulative_successes_per_title(slice8)
    print('cummulative_successes_per_title')
    Number_of_games_played_per_type = af.get_frequency_per_type(slice1)
    print('Number_of_games_played_per_type')
    Time_spent_on_games_metrics = af.get_cumulative_time_spent_on_types(slice1)
    print('Time_spent_on_games_metrics')
    time_spent_on_diffrent_worlds = af.get_time_spent_on_diffrent_worlds(slice1)
    print('time_spent_on_diffrent_worlds')
    Level_reached = af.substract_level(slice1)
    print('Level_reached')
    world_time_gametitles_dummies = af.create_world_time_assesstitle_Dummies(data)
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