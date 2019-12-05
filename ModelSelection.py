# import libraries
import numpy as np
import pandas as pd
import shap
from auxiliary_functions import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns;sns.set()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# load data
import os
os.getcwd()
X_train = pd.read_csv('C:\\Users\\uocff\\PycharmProjects\\Data-Science-Bowl-2019\\Train_Test\\X_train.csv')
X_test = pd.read_csv('C:\\Users\\uocff\\PycharmProjects\\Data-Science-Bowl-2019\\Train_Test\\X_test.csv')
Y_train = pd.read_csv('C:\\Users\\uocff\\PycharmProjects\\Data-Science-Bowl-2019\\Train_Test\\Y_train.csv')
Y_train = Y_train['accuracy_group'].to_numpy()
Y_test = pd.read_csv('C:\\Users\\uocff\\PycharmProjects\\Data-Science-Bowl-2019\\Train_Test\\Y_test.csv')
Y_test = Y_test['accuracy_group'].to_numpy()

# sample model - tuned rf
rf = RandomForestClassifier(n_estimators=83, n_jobs=-1, random_state=42)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
quadratic_weighted_kappa(Y_test, Y_pred)

# confusion matrix
mat = confusion_matrix(Y_pred, Y_test)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted values')
plt.ylabel('true value');

accuracy_score(Y_test, Y_pred)

from sklearn.metrics import classification_report
print(classification_report(Y_test, Y_pred))

# shapley values
shap_values = shap.TreeExplainer(rf).shap_values(X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.summary_plot(shap_values, X_train)
