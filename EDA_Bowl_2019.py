# The Mosquitos Data Science Team
# Exploratory Data Analysis

# import libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import json
from auxiliary_functions import *
import datetime as dt

# check working directories
import os
os.getcwd()
# 'C:\\Users\\uocff\\PycharmProjects\\Data-Science-Bowl-2019'

# get data and mappings
# train test data
data_train = pd.read_csv('C:\\Kaggle\\train.csv')
data_test = pd.read_csv('C:\\Kaggle\\test.csv')

# mappings
data_specs = pd.read_csv('C:\\Kaggle\\specs.csv')

# first view
data_test.head()
data_train.head()

# shapes and types # just an exmaple function
compare_shapes(data_test, data_train)



data_train.describe()
data_train.dtypes

data_test.describe()
data_test.dtypes

data_test.isnull().sum()
data_train.isnull().sum()

# lollipop charts for proportions in categorical columns
self_proportion_plot(data_test, 'type')
self_proportion_plot(data_test, 'world')
self_proportion_plot(data_test, 'title')


# unique values
data_train.nunique()
# or
for column in data_train.columns.values:
    print("Unique values of", column, "----", data_train[column].nunique())

# split json columns
event_data_train = pd.io.json.json_normalize(data_train['event_data'].apply(json.loads))
event_data_test = pd.io.json.json_normalize(data_test['event_data'].apply(json.loads))

# unsplit event info
specs_unsplit = pd.DataFrame()
for i in range(0, data_specs.shape[0]):
    for j in json.loads(data_specs.args[i]):
        new_df = pd.DataFrame({'event_id': data_specs['event_id'][i],
                               'info': data_specs['info'][i],
                               'args_name': j['name'],
                               'args_type': j['type'],
                               'args_info': j['info']}, index=[i])
        specs_unsplit = specs_unsplit.append(new_df)

specs_unsplit.describe()
specs_unsplit.isnull()
specs_unsplit.head()


#merge specs to train and test data
data_train_spec = pd.merge(data_train, specs_unsplit, on='event_id')
data_test_spec = pd.merge(data_test, specs_unsplit, on='event_id')

# create date time vars based on timestamp
convert_datetime(data_train)
convert_datetime(data_test)


