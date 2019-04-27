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
# max_evals = 100
# init_learning_rate = .025

iteration = 0
best_score = 0
best_iteration = 0
train_score = 0
scale = 0

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


def objective(simfin, eval_metric, space):
    global iteration, best_iteration, best_score, train_score
    iteration += 1
    log.info(f'Iteration: {iteration} ...')
    model = CatBoostClassifier(**space, random_state=123)
    model.fit(
        simfin.X_train_split, simfin.y_train_split,
        eval_set=(simfin.X_val_split, simfin.y_val_split)
    )
    score = model.best_score_['validation_0'][eval_metric]
    if score > best_score:
        best_score = score
        best_iteration = model.best_iteration_
        train_score = model.best_score_['learn'][eval_metric]
        log.info(f'Best score: {best_score} Params: {space}')
    return{'loss': -score, 'status': STATUS_OK}


class CatboostTarget:

    def catboost_target(self, init_learning_rate=.025, max_evals=100, eval_metric="Precision", od_wait=150, verbose=0):

        global best_iteration, best_score, iteration

        scale = (len(self.y_train) - sum(self.y_train)) / len(self.y_train)

        log.info("Modelling target with tuned catboost parameters...")

        # Parameter space
        space = {
            'n_estimators': 100000,
            'learning_rate': init_learning_rate,
            'task_type': 'GPU',
            'scale_pos_weight': scale,
            'eval_metric': eval_metric,
            'od_type': 'Iter',
            'od_wait': od_wait,
            'verbose': verbose,
        }

        # Get best score using default parameters
        model = CatBoostClassifier(**space, random_state=123)
        model.fit(
            self.X_train_split, self.y_train_split,
            eval_set=(self.X_val_split, self.y_val_split)
        )
        default_score = model.best_score_['validation_0'][eval_metric]
        log.info(f"Not tuned CV score: {default_score}")

        space_tune = {**space,
                      **{
                          'depth': hp.quniform('depth', 6, 10, 1),
                          'l2_leaf_reg': hp.quniform('l2_leaf_reg', 2, 30, 1),
                          'random_strength': hp.loguniform('random_strength', np.log(1), np.log(20)),
                      }
        }

        trials = Trials()
        # Tune parameters other than learning rate and estimators
        log.info("Start initial tuning ...")
        best = fmin(fn=partial(objective, self, eval_metric), space=space_tune, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=0,
                    show_progressbar=False)

        if default_score > best_score:
            log.info("Using default parameters ...")
            space_tune = space

        # Decrease learning rate to improve quality while keeping high estimators
        space_tune = {**space_tune, **best,
                 **{
                     'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(init_learning_rate)),
                 }
        }

        best_score = 0
        iteration = 0
        trials = Trials()
        log.info("Start final tuning ...")
        best = fmin(fn=partial(objective, self, eval_metric), space=space_tune, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=0,
                    show_progressbar=False)


        params = {**space_tune, **best, **{
                     'n_estimators': best_iteration,
                 }
        }

        model = CatBoostClassifier(**params)
        # Refit model with tuned parameters
        log.info(f"Final parameters: {params}")
        log.info("Refitting model with tuned hyper parameters ...")
        model.fit(self.X_train, self.y_train, verbose=verbose)

        y_pred = model.predict(self.X_test)
        test_score = precision_score(self.y_test, y_pred)

        log.info(f'Mean train score: {train_score}')
        log.info(f'Mean validation score: {best_score}')
        log.info(f'Test score: {test_score}')



