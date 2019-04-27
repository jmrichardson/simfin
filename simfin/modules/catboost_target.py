from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import precision_score
from catboost import CatBoostClassifier
from loguru import logger as log
import numpy as np
import pandas as pd
from functools import partial


# Init variables
# Number of hyper opt probes/iterations
max_evals = 10
init_learning_rate = .025


iteration = 0
best_score = 0
best_iteration = 0
train_score = 0
scale = 0

# X_train = pd.DataFrame()
# X_test = pd.DataFrame()
# y_train = pd.DataFrame()
# y_test = pd.DataFrame()
# groups = pd.Series()


# # Get cross validation score
# def validation_score(space):
#     global train_cv_score
#     model = CatBoostClassifier(**space)
#     cv = GroupKFold(n_splits=5)
#     scores = cross_validate(model, X_train, y_train, cv=cv,
#                             groups=groups, scoring='precision', n_jobs=n_jobs, return_train_score=True)
#     train_cv_score = scores['train_score'].mean()
#     test_score = scores['test_score'].mean()
#     log.info(f'Train score: {train_cv_score}  Validation score: {test_score}')
#     return test_score


def objective(simfin, space):
    global iteration, best_iteration, best_score, train_score
    iteration += 1
    log.info(f'Iteration: {iteration} ...')
    model = CatBoostClassifier(**space)
    model.fit(
        simfin.X_train_split, simfin.y_train_split,
        eval_set=(simfin.X_val_split, simfin.y_val_split)
    )
    score = model.best_score_['validation_0']['Precision']
    if score > best_score:
        best_score = score
        best_iteration = model.best_iteration_
        train_score = model.best_score_['learn']['Precision']
        log.info(f'Best score: {best_score} Params: {space}')
    return{'loss': -score, 'status': STATUS_OK}


class CatboostTarget:

    def catboost_target(self):

        global best_iteration, best_score, iteration

        scale = (len(self.y_train) - sum(self.y_train)) / len(self.y_train)

        log.info("Modelling target with tuned catboost ...")

        # Parameter space
        space = {
            'n_estimators': 100000,
            'learning_rate': init_learning_rate,
            'task_type': 'GPU',
            'scale_pos_weight': scale,
            'eval_metric': 'Precision',
            'od_type': 'Iter',
            'od_wait': 50,
            'verbose': 0,
            'depth': hp.quniform('depth', 6, 10, 1),
            'l2_leaf_reg': hp.quniform('l2_leaf_reg', 2, 30, 1),
            'random_strength': hp.loguniform('random_strength', np.log(1), np.log(20)),
        }

        trials = Trials()
        # Tune parameters other than learning rate and estimators
        log.info("Start initial tuning ...")
        best = fmin(fn=partial(objective, self), space=space, algo=tpe.suggest, max_evals=30, trials=trials, verbose=0,
                    show_progressbar=False)
        best_score = 0
        iteration = 0


        # Decrease learning rate to improve quality while keeping high estimators
        space = {**space, **best,
                 **{
                     'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(init_learning_rate)),
                 }
        }

        trials = Trials()
        log.info("Start final tuning ...")
        best = fmin(fn=partial(objective, self), space=space, algo=tpe.suggest, max_evals=30, trials=trials, verbose=0,
                    show_progressbar=False)


        params = {**space, **best, **{
                     'n_estimators': best_iteration,
                 }
        }
        model = CatBoostClassifier(**params)

        # Refit model with hyperopt parameters
        log.info(f"Final parameters: {params}")
        log.info("Refitting model with tuned hyper parameters ...")
        model.fit(self.X_train, self.y_train, verbose=0)

        y_pred = model.predict(self.X_test)
        test_score = precision_score(self.y_test, y_pred)

        log.info(f'Mean train score: {train_score}')
        log.info(f'Mean validation score: {best_score}')
        log.info(f'Test score: {test_score}')


