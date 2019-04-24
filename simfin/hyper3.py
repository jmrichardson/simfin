from hyperband import HyperbandSearchCV
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from loguru import logger as log

scale = (len(y_train) - sum(y_train)) / len(y_train)

estimator = CatBoostClassifier(
    eval_metric="F1",
    scale_pos_weight=scale,
)
param_dist = {
    # 'learning_rate': uniform(1e-7, 1),
    'max_depth': randint(2, 15),
    # 'scale_pos_weight': scale,
    'colsample_bylevel': uniform(0.4, 0.5),
    # 'subsample': uniform(0.4, 0.5),
    # 'reg_lambda': randint(1, 99),
    # 'gradient_iterations': randint(1, 10),
    # 'bagging_temperature': uniform(0, 1),
    # 'random_strength': randint(1, 19),
    # 'loss_function': ['Logloss', 'CrossEntropy'],
    # 'border_count': randint(1, 255),
}

cv = GroupKFold(n_splits=5)
search = HyperbandSearchCV(estimator, param_dist, cv=cv,
                           resource_param='iterations',
                           # min_iter=10,
                           max_iter=1200,
                           # scoring=('accuracy', 'roc_auc', 'balanced_accuracy', 'precision', 'recall', 'f1'),
                           # scoring='roc_auc',
                           # refit='roc_auc',
                           return_train_score=True)



log.info("Starting")
search.fit(X_train, y_train, groups=groups)
log.info("Done")
















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


