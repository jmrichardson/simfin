from hyperband import HyperbandSearchCV
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

scale = (len(y_train) - sum(y_train)) / len(y_train)
estimator = XGBClassifier(scale_pos_weight=scale, verbosity=2, seed=1)
param_dist = {
    'max_depth': randint(5, 45),
    'learning_rate': uniform(0.01, 0.5),
    'subsample': uniform(0.3, 0.6),
    'colsample_bytree': uniform(0.3, 0.6),
    'min_child_weight': randint(1, 6),
    'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    "tree_method":'gpu_hist',
    "predictor":'gpu_predictor',
}


inner_cv = GroupKFold(n_splits=10)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=1)

search = HyperbandSearchCV(estimator, param_dist, cv=inner_cv,
                           resource_param='n_estimators',
                           scoring='accuracy', return_train_score=True)

# search.fit(X_train, y_train, groups=groups)

cross_val_score(search, X=X_train, y=y_train, groups=groups, cv=outer_cv)

print(search.best_params_)

print(mean(search.cv_results_['mean_test_score']))
print(mean(search.cv_results_['mean_train_score']))