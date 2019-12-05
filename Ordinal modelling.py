import pandas as pd
import numpy as np
import auxiliary_functions as af

X_train = pd.read_csv('Data/X_train.csv')
X_test = pd.read_csv('Data/X_test.csv')
Y_train = pd.read_csv('Data/Y_train.csv')
Y_train = Y_train['accuracy_group'].to_numpy()
Y_test = pd.read_csv('Data/Y_test.csv')
Y_test = Y_test['accuracy_group'].to_numpy()

from sklearn.base import clone


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


from sklearn.tree import DecisionTreeClassifier
clf = OrdinalClassifier(DecisionTreeClassifier(max_depth=3))

clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
af.quadratic_weighted_kappa(Y_test, Y_pred)

from sklearn.ensemble import RandomForestClassifier
clf2 = OrdinalClassifier(RandomForestClassifier(n_estimators=83, n_jobs=-1, random_state=42))
clf2.fit(X_train, Y_train)
Y_pred = clf2.predict(X_test)
af.quadratic_weighted_kappa(Y_test, Y_pred)
