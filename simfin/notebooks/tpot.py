from tpot import TPOTClassifier
from sklearn.model_selection import GroupKFold
import pandas as pd
from deap import creator
from tpot.export_utils import generate_pipeline_code, expr_to_tree
import time
import numpy as np
from loguru import logger as log


sf = pd.read_pickle('tmp/rf.pkl')
df = sf.data_df

# Get rows where target is not null
# df = df[df['Target'].notnull()]

# Get X: Drop date, ticker and target
groups = df['Ticker']

X = df.sort_values(by='Date').drop(['Date', 'Ticker'], axis=1)
X = X.filter(regex=r'^(?!Target).*$')

# Get y
y = df.filter(regex=r'Target.*').values.ravel()

# gkf = GroupKFold(n_splits=4).split(X=X, y=y, groups=groups)
# gkf = GroupKFold(n_splits=4)

# tpot_config = {
    # 'sklearn.ensemble.RandomForestClassifier',
    # 'sklearn.tree.ExtraTreeClassifier',
# }

tpot_config = {

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [10, 50, 100],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [1, 5, 10, 20],
        'max_depth': [5, 10, 15, 20, None],
    },

    'xgboost.XGBClassifier': {
        'n_estimators': [10, 50, 100],
        'max_depth': [5, 10, 15, 20, None],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
    },

}

# Train model
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=3,
                     cv=GroupKFold(n_splits=5),
                     periodic_checkpoint_folder='tmp',
                     scoring='accuracy',
                     early_stop=5,
                     random_state=1,
                     memory='tmp',
                     warm_start=True,
                     config_dict=tpot_config,
                     )

# tpot.fit(X, y)
start_time = time.time()
tpot.fit(X, y, groups=groups)

print("--- %s seconds ---" % (time.time() - start_time))

print(tpot.score(X, y))

tpot.export('model.py')

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






