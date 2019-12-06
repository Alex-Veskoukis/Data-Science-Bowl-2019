import auxiliary_functions as af
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import xgboost as xgb
from sklearn.metrics import confusion_matrix
import seaborn as sns

classifiers = []

model1 = xgb.XGBClassifier()
classifiers.append(model1)

model2 = af.OrdinalClassifier(xgb.XGBClassifier())
classifiers.append(model2)

model3 = RandomForestClassifier(n_estimators=83, n_jobs=-1, random_state=42)
classifiers.append(model3)

model4 = af.OrdinalClassifier(RandomForestClassifier(n_estimators=83, n_jobs=-1, random_state=42))
classifiers.append(model4)

model5 = EnsembleVoteClassifier(clfs=[model1, model3], weights=[1, 1], refit=False)
classifiers.append(model5)

model6 = EnsembleVoteClassifier(clfs=[model1, model2, model3, model4], weights=[1, 1, 1, 1], refit=False)
classifiers.append(model6)

kappa = []
for clf in classifiers:
    print(clf)
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    kappa.append(af.quadratic_weighted_kappa(Y_test, Y_pred))

# Run RF classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=83, n_jobs=-1, random_state=42)
rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)
af.quadratic_weighted_kappa(Y_test, Y_pred)

# importance
importances = rf.feature_importances_
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

# load the iris datasets
# fit an Extra Trees model to the data
model = ExtraTreesClassifier()
model.fit(X_train, Y_train)
# display the relative importance of each attribute
print(model.feature_importances_)

# RFE
from sklearn.feature_selection import RFE

rfe = RFE(rf, 10)
rfe = rfe.fit(X_train, Y_train)
# summarize the selection of the attributes
print(rfe.support_)
print(rfe.ranking_)

# Tune rf
n_estimators = range(1, 100)
train_results = []
test_results = []
for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1, random_state=42)
    rf.fit(X_train, Y_train)
    train_pred = rf.predict(X_train)
    y_pred = rf.predict(X_test)
    test_results.append(af.quadratic_weighted_kappa(Y_test, y_pred))
    train_results.append(af.quadratic_weighted_kappa(Y_train, train_pred))

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(n_estimators, train_results, 'b', label='Train kappa')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test kappa')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Kappa score')
plt.xlabel('n_estimators')
plt.show()
plt.close()

# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import  make_scorer
# rf_params = {
#     'n_estimators': range(5,100),
#     'max_features': ['auto', 'sqrt', 'log2'],
# }
#
# gs_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_params, cv= 5, n_iter=60, scoring=make_scorer(quadratic_weighted_kappa, greater_is_better=True))
#
# gs_random.fit(X_train, Y_train)
#
# print(gs_random.best_params_)

# voting classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# clf1 = MLPClassifier(solver='lbfgs', alpha=1e-5,
#                      hidden_layer_sizes=(5, 2), random_state=1)
# eclf1 = clf1.fit(X_train, Y_train)
# Y_pred1 = eclf1.predict(X_test)
# af.quadratic_weighted_kappa(Y_test, Y_pred1)

clf2 = RandomForestClassifier(n_estimators=33, n_jobs=-1, random_state=42)
eclf2 = clf2.fit(X_train, Y_train)
Y_pred2 = eclf2.predict(X_test)
af.quadratic_weighted_kappa(Y_test, Y_pred2)

clf3 = VotingClassifier(estimators=[('mlp', clf1), ('rf', clf2)],
                        voting='hard')
eclf3 = clf3.fit(X_train, Y_train)
Y_pred3 = eclf3.predict(X_test)
af.quadratic_weighted_kappa(Y_test, Y_pred3)



sns.set()

mat = confusion_matrix(Y_pred, Y_test)
sns.heatmap(mat, square=True, annot=True, cbar=False)
plt.xlabel('predicted values')
plt.ylabel('true value');

af.mat(X_train, Y_train)

X_train.to_numpy
