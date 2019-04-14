import talib
import numpy as np
from loguru import logger as log
import fastai.tabular
from config import *


# Add momentum feature on features
def mom(df, count):
    global key_features
    for feature in key_features:
        if df[feature].isnull().all():
            continue
        for i in range(1, count + 1, 1):
            try:
                df['Feature_Mom_' + str(i) + 'Q_' + feature] = talib.MOM(np.array(df[feature]), i)
            except Exception as e:
                log.warning("Momentum " + str(feature) + ": " + str(e))
    return df


# Calculate trailing twelve months
def TTM(df, roll, days):

    global key_features

    def lastYearSum(series):
        # Must have x previous inputs
        if len(series) < roll:
            return np.nan
        # Must be within X days date range
        else:
            firstDate = df['Date'][series.head(1).index.item()]
            lastDate = df['Date'][series.tail(1).index.item()]
            if (lastDate - firstDate).days > days:
                return np.nan
            else:
                return series.sum()

    for feature in key_features:
        if df[feature].isnull().all():
            continue
        try:
            df['Feature_TTM_' + feature] = df[feature].rolling(roll, min_periods=1).apply(lastYearSum, raw=False)
        except:
            log.warning("Unable to add TTM for: " + feature)
    return df


# Process data by ticker
def by_ticker(df):
    global key_features
    log.info("Processing " + str(df['Ticker'].iloc[0]) + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Trailing twelve month
    df = TTM(df, 4, 370)

    # Momentum
    df = mom(df, 6)
    return df


class Features:
    def features(self):
        log.info("Add features by ticker ...")
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker)

        # Add date information
        fastai.tabular.add_datepart(df=self.data_df, field_name='Date', prefix='Feature_', drop=False)
        self.data_df['Feature_Quarter'] = self.data_df['Date'].dt.quarter
        self.data_df.reset_index(drop=True, inplace=True)
        return self


