# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:20:53 2019

@author: uocvc
"""


import os
directory = '\\\\grathfsw120p\\Roaming$\\uocvc\\Desktop\\Python\\Data-Science-Bowl-2019\\Data'
os.chdir(directory)
import pandas as pd

#=============================================================================
 # # Uncomment to display all columns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#=============================================================================
train = pd.read_csv('train.csv')
train_labels = pd.read_csv('train_labels.csv')

train.head()
train_labels.head()

train_Assessments = train.loc[train.type=='Assessment']


train_Assessments_Full = pd.merge(train_Assessments, train_labels, how='left', on=['installation_id', 'title', 'game_session'])

train_Assessments_Full.head()
