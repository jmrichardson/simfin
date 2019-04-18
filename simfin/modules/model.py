import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from loguru import logger as log
from time import time
from statistics import mean
from tpot import TPOTClassifier
from sklearn.model_selection import GroupKFold
import pandas as pd
from deap import creator
from tpot.export_utils import generate_pipeline_code, expr_to_tree
import time
import numpy as np
from loguru import logger as log
from xgboost import XGBClassifier


class Model:

    def model(self):

        log.info(f"Generating model ...")

        grid = {
            'n_estimators': [5, 10],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': [1, 5, 10],
            'max_depth': [10, None],
        }

        dist = {
            'n_estimators': sp_randint(1, 100),
            'max_depth': sp_randint(5, 40),
            'subsample': np.arange(0.05, 1.01, 0.05),
        },


        dist = {
            'n_estimators': sp_randint(1, 100),
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_leaf': sp_randint(1, 10),
            'max_depth': sp_randint(5, 40),
        }

        tpot_config = {

            'sklearn.ensemble.RandomForestClassifier': {
                'n_estimators': [10, 50, 100],
                'max_features': ['auto', 'sqrt', 'log2'],
                'min_samples_leaf': [1, 5, 10, 20],
                'max_depth': [5, 10, 15, 20, None],
            },

        }

        # Train model
        tpot = TPOTClassifier(generations=3, population_size=10, verbosity=3,
                              cv=GroupKFold(n_splits=5),
                              periodic_checkpoint_folder='tmp',
                              scoring='accuracy',
                              early_stop=4,
                              random_state=1,
                              config_dict=tpot_config,
                              )

        start_time = time.time()
        tpot.fit(X_train, y_train, groups=groups)
        print("--- %s seconds ---" % (time.time() - start_time))

        print(tpot.score(X_train, y_train))

        tpot.export('model.py')




        estimator = RandomForestClassifier(n_jobs=-1, random_state=1)
        estimator = XGBClassifier()
        cv = GroupKFold(n_splits=3)

        # search = GridSearchCV(estimator, param_grid=grid, cv=cv)
        search = RandomizedSearchCV(estimator, param_distributions=dist, cv=cv, n_iter=5,
                                    scoring=('accuracy', 'roc_auc'), refit='accuracy',
                                    return_train_score=True)

        search = RandomizedSearchCV(estimator, param_distributions=dist, cv=cv, n_iter=5,
                                    scoring=('accuracy', 'roc_auc'), refit='accuracy',
                                    return_train_score=True)


        start = time()
        model = search.fit(X_train, y_train, groups=groups)
        end = time()-start

        print(f"Took {end} seconds")
        print((model.best_params_, model.best_score_))
        print(mean(model.cv_results_['mean_test_accuracy']))
        print(mean(model.cv_results_['mean_train_accuracy']))




        y_pred = model.predict(X_train)

        y_pred = tpot.predict(X_train)

        classification_report(y_train, y_pred, target_names=['Down', 'Up'])








        return self


