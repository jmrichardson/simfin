from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import precision_score
from catboost import CatBoostClassifier
from loguru import logger as log
import numpy as np
import pandas as pd

# Init variables
# Number of hyper opt probes/iterations
max_evals = 1
n_jobs = 1

iteration = 0
best_cv_score = 0
train_cv_score = 0
X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
groups = pd.Series()

# Parameter space
space = {
    'eval_metric': 'Precision',
    'task_type': 'GPU',
    'verbose': 0,
    # 'n_estimators': 3,
    # 'n_estimators': hp.quniform('n_estimators', 10, 20, 1),
    # 'depth': hp.quniform('depth', 6, 10, 1),
    # 'learning_rate': hp.loguniform('learning_rate', np.log(0.001), np.log(.1)),
    # 'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 50, 1),
    # 'bagging_temperature': hp.quniform('bagging_temperature', 0.0, 100, 1),
    # 'random_strength': hp.quniform('random_strength', 0.0, 100, 1),
    # 'logging_level' : 'Verbose',
    # 'metric_period' : 10,
    # 'max_bin' : hp.quniform('max_bin', 1, 32, 1), #if using CPU just set this to 254
    # 'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 1),
    # 'random_strength' : hp.loguniform('random_strength', np.log(0.005), np.log(5)),
    # 'one_hot_max_size' : hp.quniform('one_hot_max_size', 2, 16, 1), #uncomment if using categorical features
    # 'bootstrap_type' : hp.choice('bootstrap_type', bootstrap_type),
    # 'eval_metric' : hp.choice('eval_metric', eval_metric_list),
    # 'objective' : OBJECTIVE_CB_REG,
    # 'score_function' : hp.choice('score_function', score_function),
    # 'leaf_estimation_backtracking' : hp.choice('leaf_estimation_backtracking', LEB),
    # 'grow_policy': hp.choice('grow_policy', grow_policy),
    # #'colsample_bylevel' : hp.quniform('colsample_bylevel', 0.1, 1, 0.01),# CPU only
    # 'fold_len_multiplier' : hp.loguniform('fold_len_multiplier', np.log(1.01), np.log(2.5)),
    # 'od_type' : 'Iter',
    # 'od_wait' : 25,
}


# Get cross validation score
def validation_score(space):
    global train_cv_score
    model = CatBoostClassifier(**space)
    cv = GroupKFold(n_splits=5)
    scores = cross_validate(model, X_train, y_train, cv=cv,
                            groups=groups, scoring='precision', n_jobs=n_jobs, return_train_score=True)
    train_cv_score = scores['train_score'].mean()
    test_score = scores['test_score'].mean()
    log.info(f'Train score: {train_cv_score}  Validation score: {test_score}')
    return test_score


# Hyperopt minimization function
def objective(space):
    global iteration, best_cv_score
    iteration += 1
    log.info(f'Starting iteration: {iteration} of {max_evals} ...')
    score = validation_score(space)
    if score > best_cv_score:
        best_cv_score = score
        log.info(f'Best score: {best_cv_score} params: {space}')
    return{'loss': -score, 'status': STATUS_OK }


class CatboostTarget:

    def catboost_target(self):

        global X_train, y_train, groups
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        groups = self.groups

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=0,
                    show_progressbar=False)

        # Get parameters and apply to model
        params = {**space, **best, **{'verbose': 1}}
        model = CatBoostClassifier(**params)

        # Refit model with hyperot parameters
        log.info("Refitting model with best hyper parameters ...")
        model.fit(X_train, y_train, verbose=0)

        log.info("Model predictions on test set ...")
        y_pred = model.predict(X_test)
        test_score = precision_score(y_test, y_pred)

        log.info(f'Mean train score: {train_cv_score}')
        log.info(f'Mean validation score: {best_cv_score}')
        log.info(f'Test score: {test_score}')

