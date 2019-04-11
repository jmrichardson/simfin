import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the class is labeled 'target' in the data file

X = df.sort_values(by='Date').drop(['Date', 'Ticker'], axis=1)
X = X.filter(regex=r'^(?!Target).*$')
y = df.filter(regex=r'Target.*').values.ravel()

imputer = Imputer(strategy="median")
imputer.fit(X)
X = imputer.transform(X)

model = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    XGBClassifier(learning_rate=0.01, max_depth=5, n_estimators=50, subsample=0.7000000000000001)
)

model.fit(X, y)

results = model.predict(testing_features)

