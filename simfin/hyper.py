from hyperband import HyperbandSearchCV
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score

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
}


cv = GroupKFold(n_splits=10)

search = HyperbandSearchCV(estimator, param_dist, cv=cv, n_jobs=-1,
                           resource_param='n_estimators',
                           scoring=('accuracy', 'roc_auc', 'balanced_accuracy', 'precision', 'recall', 'f1'),
                           refit='balanced_accuracy',
                           return_train_score=True)

search.fit(X_train, y_train, groups=groups)

print(search.best_params_)
print(mean(search.cv_results_['mean_test_balanced_accuracy']))
print(mean(search.cv_results_['mean_train_balanced_accuracy']))
print(mean(search.cv_results_['mean_test_precision']))
print(mean(search.cv_results_['mean_train_precision']))
print(mean(search.cv_results_['mean_train_recall']))
print(mean(search.cv_results_['mean_test_recall']))
print(mean(search.cv_results_['mean_train_f1']))
print(mean(search.cv_results_['mean_test_f1']))
print(mean(search.cv_results_['mean_test_f1']))
print(mean(search.cv_results_['mean_train_roc_auc']))
print(mean(search.cv_results_['mean_test_roc_auc']))

y_pred = search.predict(X_test)
confusion_matrix(y_test, y_pred)
precision_score(y_test, y_pred)
roc_auc_score(y_test, y_pred)


