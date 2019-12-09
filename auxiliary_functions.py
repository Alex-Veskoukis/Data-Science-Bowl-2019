import functools


def reduce_concat(x, sep=""):
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)


def paste(*lists, sep=" ", collapse=None):
    result = map(lambda x: reduce_concat(x, sep=sep), zip(*lists))
    if collapse is not None:
        return reduce_concat(result, sep=collapse)
    return list(result)


paste0 = functools.partial(paste, sep="")


def compare_shapes(x, y):
    print(x.shape[1] == y.shape[1])


def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
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
    import numpy as np
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b, min_rating, max_rating)
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


def self_proportion_plot(data, col):
    import matplotlib.pyplot as plt
    nrows = len(data[col])
    proportions = data[col].value_counts() / nrows
    proportions = proportions.to_frame()
    proportions = proportions.reset_index()
    proportions = proportions.rename(columns={"index": "Group"})

    # Reorder it following the values:
    ordered_proportions = proportions.sort_values(ascending=True, by=col)
    my_range = range(1, len(proportions) + 1)
    plt.hlines(y=my_range, xmin=0, xmax=ordered_proportions[col], color='black')
    plt.plot(ordered_proportions[col], my_range, "o", ms=7)
    plt.yticks(my_range, ordered_proportions['Group'])
    plt.title("Proportions of {}".format(col), loc='left')
    plt.xlabel('Percentage of Group')
    plt.ylabel('Group')
    plt.show()


def convert_datetime(df):
    import pandas as pd
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['month'] = df['timestamp'].dt.month
    df['hour'] = df['timestamp'].dt.hour
    df['year'] = df['timestamp'].dt.year
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)
    ax.set(xlim=xlim, ylim=ylim)


from sklearn.base import clone
import numpy as np


class OrdinalClassifier():

    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = clone(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf

    def predict_proba(self, X):
        clfs_predict = {k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted = []
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[y][:, 1])
            elif y in clfs_predict:

                # Vi = Pr(y > Vi-1) - Pr(y > Vi)
                predicted.append(clfs_predict[y - 1][:, 1] - clfs_predict[y][:, 1])
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[y - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


import numpy as np


class VotingClassifier(object):
    """ Implements a voting classifier for pre-trained classifiers"""

    def __init__(self, estimators):
        self.estimators = estimators

    def predict(self, X):
        # get values
        Y = np.zeros([X.shape[0], len(self.estimators)], dtype=int)
        for i, clf in enumerate(self.estimators):
            Y[:, i] = clf.predict(X)
        # apply voting
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y[i] = np.argmax(np.bincount(Y[i, :]))
        return y


def get_previous_ac_metrics(data):
    import  pandas as pd
    Assess = data.sort_values(['installation_id', 'timestamp', 'game_session','Assessments_played_Counter'], ascending=[True, True, True, True]).copy()
    Assess = Assess[Assess.type == 'Assessment']
    Assess['Attempts'] = Assess.groupby(['installation_id', 'game_session'])['Attempt'].transform(np.sum)
    Assess['Success'] = Assess.groupby(['installation_id', 'game_session'])['IsAttemptSuccessful'].transform(np.sum)
    Assess = Assess[['installation_id', 'game_session', 'Assessments_played_Counter' ,'Attempts', 'Success']].drop_duplicates()
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

    pastAG = Assess[['installation_id', 'game_session', 'Assessments_played_Counter', 'Past_Assessment_ag']]
    Assess2 = pd.pivot_table(pastAG,
                             index=['installation_id', 'game_session','Assessments_played_Counter'],
                             columns='Past_Assessment_ag',
                             values= 'Past_Assessment_ag',
                             aggfunc='nunique').reset_index()
    Assess2 = Assess2.rename(columns={0: "zeros", 1: "ones", 2: "twos", 3:'threes'})
    Assess2 = Assess2.sort_values(['installation_id','Assessments_played_Counter'])
    Assess2.loc[:,[ "zeros", 'ones',"twos" , 'threes']] = Assess2.groupby('installation_id')[ "zeros", 'ones',"twos" , 'threes'].apply(np.cumsum)
    Assess2.loc[:, ["zeros", 'ones', "twos", 'threes']] = Assess2.groupby('installation_id')[ "zeros", 'ones',"twos" , 'threes'].fillna(method='ffill')
    Assess2.loc[:, ["zeros", 'ones', "twos", 'threes']] = Assess2.loc[:, ["zeros", 'ones', "twos", 'threes']].fillna(0)
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
    AssessFinal = pd.merge(Assess,Assess2, on = ['installation_id', 'game_session'], how= 'inner')
    return AssessFinal


def get_past_attemps_and_successes(data):
    slice2 = data.loc[(data.game_time == data.Total_Game_Session_Time) &
                      (data.event_count == data.Total_Game_Session_Events),
                      ['installation_id', 'game_session', 'type',
                       'Cumulative_Attempts', 'Cumulative_Successes','Cumulative_Fails',
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
    slice2 = slice2.loc[:, ['installation_id', 'game_session',
                            'Game_Session_Order', 'Past_Total_Attempts',
                            'Past_Total_Successes', 'Past_Total_Fails','Past_Assessments_Played']]
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
                                index=['installation_id', 'game_session','type', 'Game_Session_Order'],
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
        index=['installation_id', 'game_session', 'Game_Session_Order','type'],
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
    ['Time_spent_on_Activities', 'Time_spent_on_Games','Time_spent_on_Assessments']].sum(axis=1)

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
    world_slice3 =  pd.pivot_table(
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
    Assessments = Assessments[Assessments.hour == Assessments.groupby(['installation_id', 'game_session'])['hour'].transform('min')]
    return Assessments



def get_vists_per_title(data):
    import pandas as pd
    title_slice5 =pd.pivot_table(data[['installation_id', 'game_session', 'type', 'title', 'Game_Session_Order']],
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
