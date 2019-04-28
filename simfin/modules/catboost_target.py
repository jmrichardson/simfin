from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import precision_score
from catboost import CatBoostClassifier
from loguru import logger as log
import numpy as np
import pandas as pd
from functools import partial

iteration = 0
best_score = 0
best_iteration = 0
train_score = 0


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

        iteration = 0
        best_score = 0
        best_iteration = 0

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
        base_score = model.best_score_['validation_0'][eval_metric]
        base_train_score = model.best_score_['learn']['Precision']
        base_iteration = model.best_iteration_
        if base_iteration == 0:
            base_iteration = 1
        log.info(f"Base iteration: {base_iteration}")
        log.info(f"Base validation score: {base_score}")

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

        log.info(f"best score: {best_score}")
        if best_score > base_score:
            log.info("Using tuned parameters ...")
            space = {**space, **best}
            base_score = best_score
            base_train_score = train_score
            base_iteration = best_iteration
            log.info(f"Base2 iteration: {base_iteration}")

        space_learn = {**space,
                         **{
                            'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(init_learning_rate)),
                           }
                       }

        best_score = 0
        iteration = 0
        trials = Trials()
        log.info("Start final tuning ...")
        best = fmin(fn=partial(objective, self, eval_metric), space=space_learn, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=0,
                    show_progressbar=False)

        if best_score > base_score:
            log.info("Using tuned parameters ...")
            space = {**space, **best}
            base_score = best_score
            base_iteration = best_iteration
            log.info(f"Base3 iteration: {base_iteration}")
            base_train_score = train_score

        params = {**space, **{
                               'n_estimators': base_iteration,
                             }
                  }

        log.info(f"Final parameters: {params}")
        self.catboost_target_params = params

        log.info("KFold cross validation ...")
        model = CatBoostClassifier(**params, random_state=123)
        cv = GroupKFold(n_splits=5)
        scores = cross_validate(model, self.X_train, self.y_train, cv=cv,
                                groups=self.groups, scoring='precision', return_train_score=True)
        train_cv_score = scores['train_score'].mean()
        train_cv_scores = scores['train_score']
        test_cv_score = scores['test_score'].mean()
        test_cv_scores = scores['test_score']

        log.info("Refitting model with tuned hyper parameters ...")
        model = CatBoostClassifier(**params, random_state=123)
        model.fit(self.X_train, self.y_train, verbose=verbose)

        y_pred = model.predict(self.X_test)
        test_score = precision_score(self.y_test, y_pred)

        log.info(f'Training score: {base_train_score}')
        log.info(f'Validation score: {base_score}')
        log.info(f'Training KFold scores: {train_cv_scores}')
        log.info(f'Validation KFold scores: {test_cv_scores}')
        log.info(f'Training KFold score: {train_cv_score}')
        log.info(f'Validation KFold score: {test_cv_score}')
        log.info(f'Test score: {test_score}')

        self.data_df['Predict_Catboost_Target'] = model.predict(self.X)

        return self


