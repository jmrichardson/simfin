from loguru import logger as log
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np


# Check for missing quarters, insert null row and column
def by_ticker(df):

    ticker = str(df['Ticker'].iloc[0])
    log.info("Processing {} ...".format(ticker))

    # Need to reset index to allow for concat below
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Remove rows with empty target
    # df = df[df[[c for c in df if c.startswith('Target')]].notnull().iloc[:, 0]]

    index = df.loc[:, ['Date', 'Ticker']]
    X = df.drop(['Date', 'Ticker'], axis=1)

    X = X.filter(regex=r'^(?!Target).*$')
    y = df.filter(regex=r'^Target$')

    col_names = X.columns

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X)
    X = imputer.transform(X)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X = X.astype(np.float64)
    X = pd.DataFrame(X)
    X.columns = col_names

    df = pd.concat([index, X, y], axis=1)

    return df


class Process:
    def process(self):
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker)
        self.data_df.reset_index(drop=True, inplace=True)

        return self


