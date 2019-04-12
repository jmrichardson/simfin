import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.impute import SimpleImputer
from target_reg import *
from config import *
from loguru import logger as log


def predict_rf_reg_feature(df, lag, max_depth, max_features, min_samples_leaf, n_estimators):

    global key_features

    for feature in key_features:
        log.info(f"Feature {feature} ...")

        X = df.drop(['Date', 'Ticker'], axis=1)
        train = target_reg_by_ticker(df, field=feature, lag=lag)
        train = train[pd.notnull(train['Target'])]
        y_train = train.filter(regex=r'Target.*').values.ravel()
        X_train = train.filter(regex=r'^(?!Target).*$')
        # groups = X_train['Ticker']
        X_train = X_train.drop(['Date', 'Ticker'], axis=1)

        # Impute nans
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)

        rf = RandomForestRegressor(max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf,
                                       n_estimators=n_estimators)

        # scores = cross_validate(rf, X_train, y_train, cv=GroupKFold(n_splits=5), groups=groups, return_train_score=True)

        log.info("Fitting model ...")
        rf.fit(X_train, y_train)

        X = imputer.transform(X)

        df['Predict_rf_reg_' + feature] = pd.Series(rf.predict(X))

    return df


