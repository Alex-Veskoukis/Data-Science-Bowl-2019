# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:20:53 2019
@author: uocvc
"""


# =============================================================================
# import os
# directory = 'C:/Users/Alex/Desktop/data-science-bowl-2019/Data'
# os.chdir(directory)
# =============================================================================
import pandas as pd
import numpy as np
#from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import statsmodels.api as sm


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


def Manipulate_Baseline_Data(data):
    Assessments = data[data.type=='Assessment'].copy()
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
    Assessments = Assessments.sort_values('timestamp', ascending = True)
    Assessments= Assessments[Assessments.Attempt == 1]
    Model_Base = Assessments.drop_duplicates()
    Model_Base['PastGames'] =     Model_Base.groupby('installation_id')['game_session'].transform(lambda x: pd.factorize(x)[0])
    Model_Base = Model_Base.sort_values(['installation_id','timestamp'], ascending = [True,True])
#    (Model_Base.groupby('installation_id')['game_session'].transform('cumcount')) + 1
    #Model_Base.groupby('installation_id')['game_session'].cumcount()+1
    #(Model_Base.groupby('installation_id')['game_session'].transform('cumcount')) + 1
    Model_Base['Attempts'] = Model_Base.groupby(['installation_id','game_session'])['Attempt'].transform(np.sum)
    Model_Base['Success']=Model_Base.groupby(['installation_id','game_session'])['IsAttemptSuccessful'].transform(np.sum)
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
    Model_Base= Model_Base[['Attempts','Success','title','PartOfDay','world','GameSessionTime','GameEventsCount','PastGames']]
    Model_Base = pd.concat([Model_Base, pd.get_dummies(Model_Base['PartOfDay'])] ,axis=1)
    del Model_Base['PartOfDay']
    Model_Base = pd.concat([Model_Base, pd.get_dummies(Model_Base['world'])] ,axis=1)
    del Model_Base['world']
    return Model_Base.drop_duplicates()



ModelBase = Manipulate_Baseline_Data(train)

# Attach ground truth on train
train_labels['title'] = train_labels['title'].str.rstrip(' (Assessment)')
Accuracy = train_labels[['installation_id', 'title', 'game_session','accuracy_group']]
ModelBase_train = pd.merge(Accuracy, ModelBase,  how='right', on=['installation_id', 'title', 'game_session'])
ModelBase_train = pd.concat([ModelBase_train, pd.get_dummies(ModelBase_train['title'])] ,axis=1)
ModelBase_train=ModelBase_train.drop(['title'], axis = 1)

ModelBase_train = ModelBase_train.set_index(['installation_id','game_session'])
#ModelBaseWithAccuracy.to_csv('ModelBaseWithAccuracy.csv')


ModelBase_Test = Manipulate_Baseline_Data(test)
conditions = [
    (ModelBase_Test['Success'] /ModelBase_Test['Attempts']  == 1),
     (ModelBase_Test['Success'] /ModelBase_Test['Attempts']  == 0.5),
     (ModelBase_Test['Success'] /ModelBase_Test['Attempts']  < 0.5 ) & (ModelBase_Test['Success'] /ModelBase_Test['Attempts']  > 0 ),
      (ModelBase_Test['Success'] /ModelBase_Test['Attempts']  == 0)]
choices = [3, 2, 1,0]
# =============================================================================
# ModelBase_Test['accuracy_group'] = [3 if   (v == 1)
#                                       else 2 if v  == 0.5
#                                       else 1 if  (v < 0.5 ) & (v  > 0 )
#                                       else 0  for v in ModelBase_Test['Success'] / ModelBase_Test['Attempts'] ]
# =============================================================================
ModelBase_Test['accuracy_group'] = np.select(conditions, choices, default='black')
ModelBase_Test = pd.concat([ModelBase_Test, pd.get_dummies(ModelBase_Test['title'])] ,axis=1)
ModelBase_Test = ModelBase_Test.drop(['title'], axis =1)
ModelBase_Test = ModelBase_Test[ModelBase_Test.PastGames == ModelBase_Test.groupby('installation_id')['PastGames'].transform(max)]
# =============================================================================
# def regression_results(y_true, y_pred):
#     # Regression metrics
#     explained_variance=sk.metrics.explained_variance_score(y_true, y_pred)
#     mean_absolute_error=sk.metrics.mean_absolute_error(y_true, y_pred) 
#     mse=sk.metrics.mean_squared_error(y_true, y_pred) 
#     mean_squared_log_error=sk.metrics.mean_squared_log_error(y_true, y_pred)
#     median_absolute_error=sk.metrics.median_absolute_error(y_true, y_pred)
#     r2=sk.metrics.r2_score(y_true, y_pred)
# 
#     print('explained_variance: ', round(explained_variance,4))    
#     print('mean_squared_log_error: ', round(mean_squared_log_error,4))
#     print('r2: ', round(r2,4))
#     print('MAE: ', round(mean_absolute_error,4))
#     print('MSE: ', round(mse,4))
#     print('RMSE: ', round(np.sqrt(mse),4))
#     
# =============================================================================
# Einai aparadekta ta modela apla gia na paroume mia idea    
X_train = ModelBase_train.loc[:, ~ModelBase_train.columns.isin(['Success','accuracy_group','Attempts'])]
Y_train = ModelBase_train['accuracy_group']


X_test = ModelBase_Test.loc[:, ~ModelBase_Test.columns.isin(['Success','accuracy_group','Attempts'])]
Y_test = ModelBase_Test['accuracy_group']


Regressor = linear_model.LinearRegression()
Regressor.fit(X_train,Y_train)


Prediction_train = Regressor.predict(X_train)
Prediction_test1 = Regressor.predict(X_test)


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    
    return 1.0 - numerator / denominator




quadratic_weighted_kappa(Y_test, Prediction_test1)


poisson_training_results = sm.GLM(Y_train, X_train, family=sm.families.Poisson()).fit()

print(poisson_training_results.summary())

Prediction_test2 = poisson_training_results.get_prediction(X_test)
quadratic_weighted_kappa(Y_test, Prediction_test2.summary_frame()['mean'].round())
