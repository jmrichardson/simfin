from tpot import TPOTClassifier
from sklearn.model_selection import GroupKFold
import pandas as pd
from deap import creator
from tpot.export_utils import generate_pipeline_code, expr_to_tree
import time
import numpy as np
from loguru import logger as log
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# y = df.filter(regex=r'Target.*').values.ravel()

tpot_config = {

    # 'sklearn.ensemble.RandomForestClassifier': {
        # 'n_estimators': [100],
        # 'n_estimators': [10, 50, 100],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        # 'min_samples_leaf': [1, 5, 10, 20],
        # 'max_depth': [5, 10, 15, 20, None],
    # },

    # 'xgboost.XGBClassifier': {
        # 'n_estimators': [100],
        # 'max_depth': [5, 10, 15, 20, None],
        # 'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        # 'subsample': np.arange(0.05, 1.01, 0.05),
    # },

    'lightgbm.LGBMClassifier': {
        'num_leaves': [10, 30, 50, 70, 100],
        'max_depth': [10, 20, 40, 80],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    }

}

# Train model
tpot = TPOTClassifier(
    generations=15,
    population_size=40,
    verbosity=2,
    cv=GroupKFold(n_splits=5),
    n_jobs=-1,
    memory='auto',
    warm_start=True,
    periodic_checkpoint_folder='tmp',
    early_stop=10,
    scoring='precision',
    config_dict=tpot_config,
)


# tpot.fit(X, y)
log.info("Starting...")
tpot.fit(X_train, y_train, groups=groups)
log.info("Done...")



tpot.export('tpot_model.py')

# print(tpot.score(X, y))

print(dict(list(tpot.evaluated_individuals_.items())[0:3]))
print(list(tpot.evaluated_individuals_.keys())[0])

for pipeline_string in sorted(tpot.evaluated_individuals_.keys()):
    deap_pipeline = creator.Individual.from_string(pipeline_string, tpot._pset)
    sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(deap_pipeline, tpot._pset), tpot.operators)
    if sklearn_pipeline_str.count('StackingEstimator'):
        print(sklearn_pipeline_str)
        print('evaluated_pipeline_scores: {}\n'.format(tpot.evaluated_individuals_[pipeline_string]))


dep_var='Target_Flat_SPQA'
# procs = [FillMissing, Normalize]
# data = TabularDataBunch.from_df(df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)



# tpot = TPOTRegressor(generations=5, population_size=50,verbosity=2, cv=TimeSeriesSplit(n_splits=15))






