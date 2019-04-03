from tpot import TPOTRegressor
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer, MissingIndicator
from fastai.tabular import *
import numpy as np

df = df[df['Target_Flat_SPQA'].notnull()]

df = df.sort_values(by='Date').drop(['Date', 'Ticker'], axis=1)
X = df.filter(regex=r'^(?!Target_).*$')
y = df.filter(regex=r'Target_.*')



missing = MissingIndicator(missing_values=np.NaN)
missing = missing.fit(X)
missing_df = pd.DataFrame(missing.transform(X))

impute = SimpleImputer(missing_values=np.NaN, strategy='median')
impute = impute.fit(X)
impute_df = pd.DataFrame(impute.transform(X), columns=X.columns)



tfm = FillMissing(cat_names=[], cont_names=X.columns)
tfm(X)

norm = Normalize(cat_names=[], cont_names=X.columns)
norm(X)





impute = SimpleImputer()
impute.fit(X)
look = pd.DataFrame(impute.transform(X))


dep_var='Target_Flat_SPQA'
# procs = [FillMissing, Normalize]
# data = TabularDataBunch.from_df(df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)



# tpot = TPOTRegressor(generations=5, population_size=50,verbosity=2, cv=TimeSeriesSplit(n_splits=15))
tpot = TPOTRegressor(generations=5, population_size=50, verbosity=2, cv=KFold(n_splits=5, random_state=None, shuffle=False))
tpot.fit(X, y)






