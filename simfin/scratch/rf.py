from simfin import *

from sklearn.model_selection import GroupKFold, cross_validate, GridSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from loguru import logger as log
import numpy as np


sf = pd.read_pickle('tmp/simfin.pkl')
df = sf.data_df

# Get rows where target is not null
# df = df[df['Target'].notnull()]

# Get X: Drop date, ticker and target
groups = df['Ticker']

X = df.sort_values(by='Date').drop(['Date', 'Ticker'], axis=1)
X = X.filter(regex=r'^(?!Target).*$')

# Get y
y = df.filter(regex=r'Target.*').values.ravel()

# rf = RandomForestRegressor(n_estimators=10, n_jobs=-1)
rf = RandomForestClassifier(n_estimators=40, n_jobs=-1)
gkf = GroupKFold(n_splits=5)
# scores = cross_val_score(rf, X, y, cv=gkf, groups=groups)
# print(scores.mean())


scores = cross_validate(rf, X, y, cv=gkf, groups=groups, scoring='accuracy', return_train_score=True)
scores


gkf = GroupKFold(n_splits=4).split(X=X, y=y, groups=groups)

rf = RandomForestClassifier(n_jobs=-1)
grid = {
    'n_estimators': [5, 10],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_leaf': [1,5,10],
    'max_depth': [10, None],
}

m = GridSearchCV(rf, grid, cv=gkf, scoring='accuracy')
m.fit(X, y)

print("Best parameters set found on development set:")
print()
print(m.best_params_)
print()
print("Grid scores on development set:")
print()
means = m.cv_results_['mean_train_score']
stds = m.cv_results_['std_train_score']
for mean, std, params in zip(means, stds, m.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()


print(m.best_params_)
print(m.best_score_)
print(m.cv_results_['mean_train_score'])
print(split)


probs = cross_val_predict(rf, X, y, cv=gkf, groups=groups, method='predict_proba')

metrics.roc_auc_score(y, probs[:, 0])





# Perform cross validation and pull AUC for various splits
auc = []
for n in [5, 10, 20]:
    est = RandomForestClassifier(n_estimators=n, max_features=8,n_jobs=-1,
                                 oob_score=True)
    probs = cross_val_predict(est, X, y, cv=gkf, method='predict_proba')
    temp_auc = []
    for j in range(Y_one_hot.shape[1]):
        temp_auc.append(metrics.roc_auc_score(Y_one_hot[:, j], probs[:, j]))
    auc.append(temp_auc)
    print('Test AUC for {0} trees: '.format(n), temp_auc)
    print('---------------------------------------------------------')

