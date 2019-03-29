import talib
import numpy as np
from loguru import logger as log


# Add momentum indicator on features
def mom(df, features, count):
    for feature in features:
        if df[feature].isnull().all():
            continue
        for i in range(1, count + 1, 1):
            try:
                df[feature + ' Mom ' + str(i) + 'Q'] = talib.MOM(np.array(df[feature]), i)
            except Exception as e:
                log.warning("Momentum " + str(feature) + ": " + str(e))
    return df


# Calculate trailing twelve months
def TTM(df, features, roll, days):
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

    for feature in features:
        if df[feature].isnull().all():
            continue
        try:
            df[feature + ' TTM'] = df[feature].rolling(roll, min_periods=1).apply(lastYearSum, raw=False)
        except:
            log.warning("Unable to add TTM for: " + feature)
    return df


# Process data by ticker
def by_ticker(df, simfinFeatures):
    log.info("Processing " + str(df['Ticker'].iloc[0]) + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Trailing twelve month
    df = TTM(df, simfinFeatures, 4, 370)

    # Momentum
    df = mom(df, simfinFeatures, 6)
    return df


def indicators_by_ticker(df, simfinFeatures):
    df = df.groupby('Ticker').apply(by_ticker, simfinFeatures)
    df.reset_index(drop=True, inplace=True)
    return df


