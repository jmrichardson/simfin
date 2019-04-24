from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import GroupKFold, cross_val_score
from catboost import CatBoostClassifier
from loguru import logger as log
import numpy as np

iteration = 0
best_score = 0


def validation_score(space):
    model = CatBoostClassifier(**space)
    cv = GroupKFold(n_splits=5)
    return cross_val_score(model, X_train, y_train, cv=cv, groups=groups).mean()


def objective(space):
    global iteration, best_score
    iteration += 1
    score = validation_score(space)
    if score > best_score:
        best_score = score
        print(space)
    log.info(f'Iteration: {iteration}, Best score: {best_score}')
    return{'loss': -score, 'status': STATUS_OK }

space ={
    'eval_metric': 'F1',
    'thread_count': -1,
    'n_estimators': hp.quniform('n_estimators', 200, 1500, 1),
    'depth': hp.quniform('depth', 6, 10, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(.3)),
    'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 50, 1),
    'bagging_temperature': hp.quniform('bagging_temperature', 0.0, 100, 1),
    'random_strength': hp.quniform('random_strength', 0.0, 100, 1),
    'task_type': 'GPU',
    'verbose': 0,
    # 'logging_level' : 'Verbose',
    # 'metric_period' : 10,
    # 'max_bin' : hp.quniform('max_bin', 1, 32, 1), #if using CPU just set this to 254
    # 'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 1),
    # 'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),
    #'one_hot_max_size' : hp.quniform('one_hot_max_size', 2, 16, 1), #uncomment if using categorical features
    # 'bootstrap_type' : hp.choice('bootstrap_type', bootstrap_type),
    # 'eval_metric' : hp.choice('eval_metric', eval_metric_list),
    # 'objective' : OBJECTIVE_CB_REG,
    #'score_function' : hp.choice('score_function', score_function),
    # 'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', LEB),
    # 'grow_policy': hp.choice('grow_policy', grow_policy),
    # #'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),# CPU only
    # 'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),
    # 'od_type' : 'Iter',
    # 'od_wait' : 25,
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials, verbose=0,
            show_progressbar=False)

print('-'*50)
print('The best params:')
print( best )
print('\n\n')