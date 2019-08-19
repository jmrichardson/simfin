from decimal import Decimal
from loguru import logger as log
import numpy as np
import pandas as pd
np.warnings.filterwarnings('ignore')


# Fix incorrect decimal placement
def dec(field):

    median = field.median()
    mdp = str(float(median)).replace('-', '').find('.')
    # if median decimal point larger than 1, ie the median(+ or -) is greater than 1
    # Transform the value if between -1 and 1, else do nothing
    if mdp > 1:
        field = field.transform(lambda x: pow(10, mdp) *
                float(("." + str(Decimal(pow(10, str(float(x))[::-1].find('.'))
                * x)).replace(".", "")).replace(".-", "-.")) if (x > -1 and x < 1) else x)

    return field


# Remove outliers in Series
def removeOutliers(x, outlierConstant=25):
    a = np.array(x)
    upper_quartile = np.nanquantile(a, .75)
    lower_quartile = np.nanquantile(a, .25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    if IQR == 0:
        # Deal with missing row column
        a = np.where((a >= -1) & (a <= 1), a, np.nan)
    else:
        a = np.where((a >= quartileSet[0]) & (a <= quartileSet[1]), a, np.nan)
    return a


def by_ticker(df):

    ticker = str(df['Ticker'].iloc[0])
    # log.info(f"Ticker {ticker}")

    # Need to reset index to allow for concat below
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Get index columns
    index = df.loc[:, ['Date', 'Ticker']]

    # Drop index columns
    X = df.drop(['Date', 'Ticker'], axis=1)

    # Drop all ALL NaN columns
    X = X.dropna(axis=1, how='all').astype(float)

    # Remove outliers using IQR
    for col in X.columns:
        X[col] = removeOutliers(X[col], 25)

    # Remove outliers which appear to have incorrect decimal places (may never need because of IQR
    for col in X.columns:
        X[col] = X[col].transform(dec)

    return pd.concat([index, X], axis=1)


class Outliers:

    def outliers(self):
        log.info("Remove outliers ...")
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker)
        # df = data_df.groupby('Ticker').apply(by_ticker)

        self.data_df.reset_index(drop=True, inplace=True)
        return self

