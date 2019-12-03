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
    import pandas as pd
    import numpy as np
    trainTitles = data['title'].unique()
    trainTitles_sub = [item for item in trainTitles if item not in ['Bird Measurer (Assessment)']]

    AttemptIndicator = (data.type == 'Assessment') & \
                       ((data.event_code.isin([4100]) & data.title.isin(trainTitles_sub)) |
                        (data.event_code.isin([4110]) & data.title.isin(['Bird Measurer (Assessment)'])))
    data['Attempt'] = 0
    data.loc[AttemptIndicator, 'Attempt'] = 1

    FailedAttemptIndicator = data['event_data'].str.contains('false') & AttemptIndicator
    SuccessfulAttemptIndicator = data['event_data'].str.contains('true') & AttemptIndicator
    data['IsAttemptSuccessful'] = 0
    data.loc[SuccessfulAttemptIndicator, 'IsAttemptSuccessful'] = 1

    data['timestamp'] = pd.to_datetime(data['timestamp'], format="%Y-%m-%d %H:%M")

    Inst_Group = data.groupby('installation_id')
    Inst_Game_Group = data.groupby(['installation_id', 'game_session'])
    data['Total_Game_Session_Time'] = Inst_Game_Group['game_time'].transform(np.max)
    data['Total_Game_Session_Events'] = Inst_Game_Group['event_count'].transform(np.max)
    data['Past_Assessments'] = data[data.type == 'Assessment'].groupby('installation_id')['game_session'].transform(
        lambda x: pd.factorize(x)[0])

    data['Past_Assessment_Attempts'] = Inst_Game_Group['Attempt'].transform(np.sum)

    Column_Order_List = ['installation_id', 'game_session', 'timestamp', 'game_time', 'event_id', 'event_code',
                         'event_count', 'title', 'type', 'world', 'event_data', 'Attempt', 'IsAttemptSuccessful',
                         'Total_Game_Session_Time', 'Total_Game_Session_Events', 'Past_Assessment_Attempts']
    data = data[Column_Order_List]
    data = data.sort_values(['installation_id', 'timestamp', 'game_session'], ascending=[True, True, True])

    data['Past_Total_Attempts'] = Inst_Group['Attempt'].transform(np.cumsum)
    data['Past_Total_Successes'] = Inst_Group['IsAttemptSuccessful'].transform(np.nancumsum)
    # Assessments['GameAssessmentSessionTime'] = Assessments.groupby(['installation_id', 'game_session'])[
    #     'game_time'].transform(np.max)
    data['GameAssessmentSessionTime'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])['game_time'].transform(np.max)
    data['GameAssessmentEvents'] = data[data.type == 'Assessment'].groupby(['installation_id', 'game_session'])['event_count'].transform(np.max)

    # Slice 1
    slice1 = data.copy().loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type', 'title', 'world', 'Total_Game_Session_Time',
                              'Total_Game_Session_Events']].drop_duplicates()
    slice1['Order'] = slice1.groupby('installation_id')['game_session'].cumcount() + 1
    slice1['Time_Spent'] = slice1.groupby(['installation_id'])['Total_Game_Session_Time'].cumsum()

    # Slice 2
    slice2 = data.copy().loc[(data.game_time == data.Total_Game_Session_Time) &
                             (data.event_count == data.Total_Game_Session_Events),
                             ['installation_id', 'game_session', 'type',
                              'Past_Total_Attempts', 'Past_Total_Successes', 'Past_Assessment_Attempts']].drop_duplicates()
    slice2['Order'] = slice2.groupby('installation_id')['game_session'].cumcount() + 1
    slice2 = slice2.sort_values(['installation_id', 'Order'])
    slice2 = slice2[slice2.type == 'Assessment']
    slice2['Past_Total_Attempts'] = round(slice2.groupby('installation_id')['Past_Total_Attempts'].shift(1, fill_value = 0))
    slice2['Past_Total_Successes'] = round(slice2.groupby('installation_id')['Past_Total_Successes'].shift(1, fill_value=0))
    slice2['Past_Assessment_Attempts'] = round(slice2.groupby('installation_id')['Past_Assessment_Attempts'].shift(1, fill_value=0))
    slice2 = slice2.loc[:, ~slice2.columns.isin(['type', 'Order'])]

    # Slice 3
    slice3 = data.loc[data.type == 'Assessment', ['installation_id',
                                                  'game_session',
                                                  'GameAssessmentSessionTime',
                                                  'GameAssessmentEvents']].drop_duplicates()

    # Slice 1 / Type frequency Experience Measures
    type_slice = pd.pivot_table(slice1[['installation_id', 'game_session', 'type', 'Order']],
                                index=['installation_id', 'game_session', 'Order'],
                                columns='type',
                                aggfunc=len,
                                fill_value=0).reset_index().sort_values(['installation_id', 'Order'])
    type_slice['Past_Activities'] = type_slice.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice['Past_Games'] = type_slice.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice['Past_Clips'] = type_slice.groupby('installation_id')['Clip'].transform(np.cumsum)
    type_slice['Past_Assessments'] = type_slice.groupby('installation_id')['Assessment'].transform(np.cumsum)
    type_slice_assessments = type_slice[type_slice.Assessment == 1]
    type_slice_assessments = type_slice_assessments.rename(columns={'Order': 'Past_Game_Sessions'})
    type_slice_assessments = type_slice_assessments.drop(['Game', 'Clip', 'Assessment', 'Activity'], axis=1)
    type_slice_assessments['Clips'] = type_slice_assessments['Past_Clips']
    type_slice_assessments['Games'] = type_slice_assessments['Past_Games']
    type_slice_assessments['Assessments'] = type_slice_assessments['Past_Assessments']
    type_slice_assessments['Activities'] = type_slice_assessments['Past_Activities']
    type_slice_assessments = type_slice_assessments.loc[:,
                             ['installation_id', 'game_session', 'Past_Game_Sessions', 'Clips',
                              'Games', 'Assessments', 'Activities']].drop_duplicates()
    # Slice 1 / Type time spent Experience Measures
    type_slice2 = pd.pivot_table(
        slice1[['installation_id', 'game_session', 'type', 'Order', 'Total_Game_Session_Time']],
        index=['installation_id', 'game_session', 'Order'],
        columns='type',
        values='Total_Game_Session_Time',
        aggfunc=sum,
        fill_value=0).reset_index().sort_values(['installation_id', 'Order'])
    type_slice2['Activity_Time_spent'] = type_slice2.groupby('installation_id')['Activity'].transform(np.cumsum)
    type_slice2['Game_Time_spent'] = type_slice2.groupby('installation_id')['Game'].transform(np.cumsum)
    type_slice2['Clip_Time_spent'] = type_slice2.groupby('installation_id')['Clip'].transform(np.cumsum)
    type_slice2['Assessment_Time_spent'] = type_slice2.groupby('installation_id')['Assessment'].transform(np.cumsum)
    type_slice2_assessments = type_slice2[type_slice2.Assessment != 0]
    type_slice2_assessments.loc[:,'Total_Time_spent'] = type_slice2_assessments[
        ['Activity_Time_spent', 'Game_Time_spent', 'Clip_Time_spent', 'Assessment_Time_spent']].sum(axis=1)
    type_slice2_assessments = type_slice2_assessments.loc[:,
                              ['installation_id', 'game_session', 'Total_Time_spent', 'Game_Time_spent',
                               'Assessment_Time_spent', 'Activity_Time_spent']].drop_duplicates()
    MergedSlices = pd.merge(type_slice_assessments, type_slice2_assessments, on=['installation_id', 'game_session'],
                            how='inner')
    # Create Dummies
    Assessments = data[data.type == 'Assessment'].copy()
    Assessments['timestamp'] = pd.to_datetime(Assessments['timestamp'], format="%Y-%m-%d %H:%M")
    Assessments = Assessments.sort_values('timestamp', ascending=True)
    Assessments = Assessments.drop_duplicates()
    # TimeVariables = convert_datetime(Assessments)


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
    Assessments['PartOfDay'] = pd.Categorical(Assessments['PartOfDay'])
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
    FinalData = pd.merge(FinalData, slice2, how='inner',
                         on=['installation_id', 'game_session'])

    FinalData = pd.merge(FinalData, slice3, how='inner',
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
    return Assess[['accuracy', 'accuracy_group', 'installation_id', 'game_session']]


Test_set = get_test_set_accuracy(test)

Test_set_full = pd.merge(Test_Features, Test_set.loc[:, ~ Test_set.columns.isin(['accuracy'])],
                         on=['installation_id', 'game_session'], how='inner')

X_train = FinalTrain.loc[:,
          ~FinalTrain.columns.isin(['accuracy_group', 'installation_id', 'game_session', 'PartOfDay'])]
Y_train = FinalTrain['accuracy_group']

X_test = Test_set_full.loc[:, ~Test_Features.columns.isin(['accuracy_group', 'installation_id', 'game_session'])]
Y_test = Test_set_full['accuracy_group'].to_numpy(dtype=int)


from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees


rf =RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                      max_features='auto', max_leaf_nodes=None,
                      min_impurity_decrease=0.0, min_impurity_split=None,
                      min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, n_estimators=8,
                      n_jobs=None, oob_score=False, random_state=None,
                      verbose=0, warm_start=False)
#n_estimators=50 ,  random_state=42 , max_features = 'auto', bootstrap=True, criterion = 'mae'
# Train the model on training data
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)

quadratic_weighted_kappa(Y_test, Y_pred)

from sklearn.model_selection import RandomizedSearchCV
rf_params = {
    'n_estimators': range(10,100),
    'max_features': ['auto', 'sqrt', 'log2'],
}

gs_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, cv= 5, n_iter=60)

gs_random.fit(X_train, Y_train)

print(gs_random.best_params_)




Prediction_test2 = poisson_training_results.get_prediction(X_test)
quadratic_weighted_kappa(Y_test, Prediction_test2.summary_frame()['mean'].round())
