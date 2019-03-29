import talib
import pandas as pd
import numpy as np
from loguru import logger as log


def by_ticker(df, min_quarters):

    log.info("Processing " + str(df['Ticker'].iloc[0]) + "...")

    if len(df.index) < min_quarters:
        log.warn("Not enough quarter history: " + str(min_quarters))
        return pd.DataFrame

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Fill Share Price NAs with last known value
    df['Share Price'] = df['Share Price'].ffill()

    # Get Last known value (these fields are reported at different times in the quarter by simfin)
    # This will get the last value within 90 days to be used by rows with published quarter data
    df['Common Shares Outstanding'] = df['Common Shares Outstanding'] \
        .rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Avg. Basic Shares Outstanding'] = df['Avg. Basic Shares Outstanding'] \
        .rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Avg. Diluted Shares Outstanding'] = df['Avg. Diluted Shares Outstanding'] \
        .rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)

    # Add a few more ratios
    df['Basic Earnings Per Share'] = df['Net Profit'] / df['Avg. Basic Shares Outstanding'] * 1000
    df['Common Earnings Per Share'] = df['Net Profit'] / df['Common Shares Outstanding'] * 1000
    df['Diluted Earnings Per Share'] = df['Net Profit'] / df['Avg. Diluted Shares Outstanding'] * 1000
    df['Basic PE'] = df['Share Price'] / df['Basic Earnings Per Share']
    df['Common PE'] = df['Share Price'] / df['Common Earnings Per Share']
    df['Diluted PE'] = df['Share Price'] / df['Diluted Earnings Per Share']

    # Average share prices for last 30 days
    df.insert(3, column='SPQA', value=df['Share Price'].rolling(90, min_periods=1).mean())
    df.insert(4, column='SPMA', value=df['Share Price'].rolling(30, min_periods=1).mean())

    # Momentum on SPMA
    try:
        df['SPMA Mom 1M'] = talib.MOM(np.array(df['SPMA']), 30)
        df['SPMA Mom 2M'] = talib.MOM(np.array(df['SPMA']), 60)
        df['SPMA Mom 3M'] = talib.MOM(np.array(df['SPMA']), 90)
        df['SPMA Mom 6M'] = talib.MOM(np.array(df['SPMA']), 180)
        df['SPMA Mom 9M'] = talib.MOM(np.array(df['SPMA']), 270)
        df['SPMA Mom 12M'] = talib.MOM(np.array(df['SPMA']), 360)
    except:
        pass

    # Remove rows where feature is null (removes many rows)
    df = df[df['Revenues'].notnull()]
    df = df[df['Net Profit'].notnull()]

    return df


def flatten_by_ticker(df, min_quarters):
    df = df.groupby('Ticker').apply(by_ticker, min_quarters)
    df.reset_index(drop=True, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

