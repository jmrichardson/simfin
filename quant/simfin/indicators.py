import talib
import numpy as np
from loguru import logger as log


# Add momentum indicator on features
def mom(df, key_features, count):
    for feature in key_features:
        if df[feature].isnull().all():
            continue
        for i in range(1, count + 1, 1):
            try:
                df['Ind_Mom_' + str(i) + 'Q_' + feature] = talib.MOM(np.array(df[feature]), i)
            except Exception as e:
                log.warning("Momentum " + str(feature) + ": " + str(e))
    return df


# Calculate trailing twelve months
def TTM(df, key_features, roll, days):
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
            df['Ind_TTM_' + feature] = df[feature].rolling(roll, min_periods=1).apply(lastYearSum, raw=False)
        except:
            log.warning("Unable to add TTM for: " + feature)
    return df


# Process data by ticker
def by_ticker(df, key_features):
    log.info("Processing " + str(df['Ticker'].iloc[0]) + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Trailing twelve month
    df = TTM(df, key_features, 4, 370)

    # Momentum
    df = mom(df, key_features, 6)
    return df


def indicators_by_ticker(df, key_features):
    df = df.groupby('Ticker').apply(by_ticker, key_features)
    df.reset_index(drop=True, inplace=True)
    return df


