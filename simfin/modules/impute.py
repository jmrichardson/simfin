from loguru import logger as log
from sklearn.impute import MissingIndicator
import pandas as pd
import numpy as np
from fancyimpute import KNN
from config import *

# Check for missing quarters, insert null row and column
def by_ticker(df, indicate=False):

    # ticker = str(df['Ticker'].iloc[0])

    # Need to reset index to allow for concat below
    df = df.sort_values(by='Date').reset_index(drop=True)

    index = df.loc[:, ['Date', 'Ticker']]
    X = df.drop(['Date', 'Ticker'], axis=1)

    # Drop Target column if exists (impute before calling target)
    # Code still exists here to call after if decide to use it
    X = X.loc[:, X.columns != 'Target']
    # y = df.loc[:, 'Target']

    # Drop all null value columns
    X = X.dropna(axis=1, how='all').astype(float)

    # Get column names
    col_names = X.columns

    # Dataframe to track missing values
    # X_missing = pd.DataFrame()

    if missing_impute:
        missing = MissingIndicator(features='all')
        missing.fit(X)
        X_missing = pd.DataFrame(missing.transform(X)).astype(float)
        X_missing.columns = "Missing_" + col_names

    # imputer = SimpleImputer(strategy="median")
    # imputer.fit(X)
    # X = imputer.transform(X)

    if X.isnull().values.any():
        X = KNN(k=5, verbose=False).fit_transform(X)

    X = X.astype(np.float64)
    X = pd.DataFrame(X)
    X.columns = col_names

    # df = pd.concat([index, X, X_missing, y], axis=1)
    if missing_impute:
        df = pd.concat([index, X, X_missing], axis=1)
    else:
        df = pd.concat([index, X], axis=1)
    return df


class Impute:
    def impute(self, indicate=False):

        log.info("Impute features by ticker ...")

        # if 'Target' not in self.data_df.columns:
            # self.data_df['Target'] = np.nan

        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker, indicate)

        # Missing field indicators (replace NANs with number out of range
        self.data_df = self.data_df.fillna(-99999)

        # If Target in df and it had nans, replace back to nans
        # if 'Target' in self.data_df.columns:
            # self.data_df['Target'] = self.data_df['Target'].replace(-99999, np.nan)

        self.data_df.reset_index(drop=True, inplace=True)
        return self


