# The Mosquitos Data Science Team
# Exploratory Data Analysis

# import libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import json
from auxiliary_functions import *

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

# split json columns
#data_train = pd.concat([data_train, data_train['event_data'].apply(json_to_series)], axis=1)
#data_test = pd.concat([data_test, data_test['event_data'].apply(json_to_series)], axis=1)




