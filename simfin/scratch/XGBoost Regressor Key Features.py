from tpot import TPOTClassifier, TPOTRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GroupKFold
import pandas as pd
import numpy as np

import os, sys
if 'notebooks' in os.getcwd():
    os.chdir('..')
sys.path.append("..")
from simfin import *


# Extract and flaten simfin data set
if not os.path.isfile('tmp/extract.zip'):
    simfin = SimFin().extract().flatten()
else:
    simfin = SimFin().flatten()

simfin = simfin.target_reg(field='Revenues', lag=-1)

df = simfin.data_df


X = df[pd.notnull(df['Target'])]
groups = X['Ticker']
y = X.filter(regex=r'Target.*').values.ravel()
X = X.filter(regex=r'^(?!Target).*$')
X = X.drop(['Date', 'Ticker'], axis=1)


tpot_config = {
    'xgboost.XGBRegressor': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'nthread': [1]
    },
}

tpot = TPOTRegressor(generations=5, population_size=20, verbosity=3,
                     cv=GroupKFold(n_splits=5),
                     scoring='accuracy',
                     early_stop=5,
                     random_state=1,
                     config_dict=tpot_config,
                     )


tpot.fit(X, y, groups=groups)

print(tpot.score(X, y))
tpot.export('model.py')



