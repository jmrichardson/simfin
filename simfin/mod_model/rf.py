from simfin import *

from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from loguru import logger as log


df = SimFin().load('rf2').data_df

# Get rows where target is not null
# df = df[df['Target'].notnull()]

# Get X: Drop date, ticker and target
groups = df['Ticker']

X = df.sort_values(by='Date').drop(['Date', 'Ticker'], axis=1)
X = X.filter(regex=r'^(?!Target).*$')

# Get y
y = df.filter(regex=r'Target.*').values.ravel()


rf = RandomForestRegressor(n_estimators=10, n_jobs=-1)

gkf = GroupKFold(n_splits=5)


scores = cross_val_score(rf, X, y, cv=gkf, groups=groups, scoring='r2')



