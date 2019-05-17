from loguru import logger as log
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, MissingIndicator
import pandas as pd
import numpy as np


# Check for missing quarters, insert null row and column
def by_ticker(df, impute, indicate=False):

    ticker = str(df['Ticker'].iloc[0])
    # log.info("Processing {} ...".format(ticker))

    # Need to reset index to allow for concat below
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Remove rows with empty target
    # df = df[df[[c for c in df if c.startswith('Target')]].notnull().iloc[:, 0]]

    index = df.loc[:, ['Date', 'Ticker']]
    X = df.drop(['Date', 'Ticker'], axis=1)

    # X = X.filter(regex=r'^(?!Target).*$')
    X = X.loc[:, X.columns != 'Target']
    # y = df.filter(regex=r'^Target$')
    # y = df.loc[:, 'Target'].values.ravel()
    y = df.loc[:, 'Target']

    # Get original column names
    X = X.dropna(axis=1, how='all').astype(float)
    col_names = X.columns
    X_missing = pd.DataFrame()

    if impute:

        if indicate:
            missing = MissingIndicator(features='all')
            missing.fit(X)
            X_missing = pd.DataFrame(missing.transform(X)).astype(float)
            X_missing.columns = "Missing_" + col_names

        imputer = SimpleImputer(strategy="median")
        imputer.fit(X)
        X = imputer.transform(X)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X = X.astype(np.float64)
    X = pd.DataFrame(X)
    X.columns = col_names

    df = pd.concat([index, X, X_missing, y], axis=1)

    return df


class Process:
    def process(self, impute=False):

        log.info("Pre-process features by ticker ...")

        # Impute and scale
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker, impute)

        # Features per ticker which are all NAN, we can't impute, so give a value out of range
        if impute:
            self.data_df = self.data_df.fillna(-9999)
            if 'Target' in self.data_df.columns:
                self.data_df['Target'] = self.data_df['Target'].replace(-9999, np.nan)

        self.data_df.reset_index(drop=True, inplace=True)
        return self


