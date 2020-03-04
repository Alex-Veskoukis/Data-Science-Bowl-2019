import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
# load data

X = X_train
Y =Y_train
# feature extraction
model = RandomForestClassifier(n_estimators= 50, n_jobs=-1, random_state=42)
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)

# feature extraction
model = RandomForestClassifier(n_estimators= 50, n_jobs=-1, random_state=42)
model.fit(X, Y)
print(model.feature_importances_)


importance = pd.DataFrame({"feature": X.columns,
                           "importance": model.feature_importances_})


importance = importance.sort_values('importance', ascending = False )
