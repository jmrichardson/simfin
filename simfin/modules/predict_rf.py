import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.impute import SimpleImputer
from loguru import logger as log


class PredictRF:

    def predict_rf(self, field='Revenues', lag=-1, type='reg', thresh=None, max_depth=10, max_features="sqrt", min_samples_leaf=5, n_estimators=100):

        log.info(f"Predicting key features ...")

        X = self.data_df.drop(['Date', 'Ticker'], axis=1)
        print(field)
        if type == 'reg':
            train = self.target(field=field, type=type, lag=lag, thresh=thresh).data_df
        else:
            train = self.target(field=field, type=type, lag=lag, thresh=thresh).data_df


        train = train[pd.notnull(train['Target'])]

        y_train = train.filter(regex=r'Target.*').values.ravel()
        X_train = train.filter(regex=r'^(?!Target).*$')
        # groups = X_train['Ticker']
        X_train = X_train.drop(['Date', 'Ticker'], axis=1)

        # Impute nans
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)

        if type == 'reg':
            rf = RandomForestRegressor(max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf,
                                       n_estimators=n_estimators)
            # scores = cross_validate(rf, X_train, y_train, cv=GroupKFold(n_splits=5), groups=groups, return_train_score=True)
        else:
            rf = RandomForestClassifier(max_depth=max_depth,
                                       max_features=max_features,
                                       min_samples_leaf=min_samples_leaf,
                                       n_estimators=n_estimators)


        log.info("Fitting model ...")
        rf.fit(X_train, y_train)

        X = imputer.transform(X)

        self.data_df['Predict_rf_' + type + '_' + field] = pd.Series(rf.predict(X))
        self.predict_rf_df = self.data_df

        return self


