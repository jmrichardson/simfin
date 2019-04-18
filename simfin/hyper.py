from hyperband import HyperbandSearchCV
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

estimator = XGBClassifier()
param_dist = {
    'max_depth': randint(5, 45),
    'learning_rate': uniform(0.01, 0.5),
    'subsample': uniform(0.25, 0.5),
    'colsample_bytree': uniform(0.25, 0.5),
    'min_child_weight': randint(1, 6),
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    'gamma': [i/10.0 for i in range(0, 5)]
}

cv = GroupKFold(n_splits=5)
search = HyperbandSearchCV(estimator, param_dist, cv=cv,
                           resource_param='n_estimators',
                           scoring='accuracy', return_train_score=True)

fit = search.fit(X_train, y_train, groups=groups)

print(search.best_params_)