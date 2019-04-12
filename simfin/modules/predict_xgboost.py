import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Imputer
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier, XGBRegressor
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from config import *
from target_reg import *


def predict_xgboost_features(self, lag, learning_rate, max_depth, n_estimators, subsample):

    global key_features

    lag = -1
    learning_rate = 0.01
    max_depth = 5
    n_estimators = 50
    subsample = 0.7
    feature='Revenues'


    for feature in key_features:

        # df = df.drop([feature], axis=1)

        X = target_reg_by_ticker(df, field=feature, lag=lag)

        X = X[pd.notnull(X['Target'])]
        y = X.filter(regex=r'Target.*').values.ravel()

        X = X.filter(regex=r'^(?!Target).*$')
        X = X.drop(['Date', 'Ticker'], axis=1)


        imputer = Imputer(strategy="median")
        imputer.fit(X)
        X = imputer.transform(X)

        model = make_pipeline(
            make_union(
                FunctionTransformer(copy),
                FunctionTransformer(copy)
            ),
            XGBRegressor(learning_rate=learning_rate, max_depth=max_depth, n_estimators=n_estimators, subsample=subsample)
        )

        model.fit(X, y)

        df[feature] = pd.Series(model.predict(X))

    return df


