from tpot import TPOTRegressor
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer, MissingIndicator
from fastai.tabular import *
import numpy as np
import pandas as pd
from loguru import logger as log
import re
import os
import sys

# Set current working directory (except for interactive shell)
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = 'd:/projects/quant/quant/simfin'

# Extend path for local imports
os.chdir(cwd)
rootPath = re.sub(r"(.*quant).*", r"\1", cwd)
sys.path.extend([cwd, rootPath])

df = pd.read_pickle('df.pck')

# Get rows where target is not null
df = df[df['Target_Flat_SPQA'].notnull()]

# Get X: Drop date, ticker and target
# df = df.sort_values(by='Date').drop(['Date', 'Ticker'], axis=1)
X = df.filter(regex=r'^(?!Target_).*$')

# Get y
y = df.filter(regex=r'Target_.*').values.ravel()


# m = RandomForestRegressor(n_estimators=50, n_jobs=-1)
# m.fit(X, y)
# m.score(X,y)



tpot_config = {
    'sklearn.ensemble.RandomForestRegressor': {
    }
}


# Train model
# tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, cv=KFold(n_splits=5, random_state=None, shuffle=False))
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2,
                     cv=KFold(n_splits=5, random_state=None, shuffle=False),
                     periodic_checkpoint_folder='tmp',
                     early_stop=5,
                     random_state=1,
                     memory='tmp',
                     warm_start=True,
                     config_dict=tpot_config
                     )
tpot.fit(X, y)






dep_var='Target_Flat_SPQA'
# procs = [FillMissing, Normalize]
# data = TabularDataBunch.from_df(df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)



# tpot = TPOTRegressor(generations=5, population_size=50,verbosity=2, cv=TimeSeriesSplit(n_splits=15))






