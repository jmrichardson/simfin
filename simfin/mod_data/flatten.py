import talib
import pandas as pd
import numpy as np
from loguru import logger as log


def by_ticker(df):

    ticker = str(df['Ticker'].iloc[0])
    log.info("Processing {} ...".format(ticker))

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
    df['Flat_Basic Earnings Per Share'] = df['Net Profit'] / df['Avg. Basic Shares Outstanding'] * 1000
    df['Flat_Common Earnings Per Share'] = df['Net Profit'] / df['Common Shares Outstanding'] * 1000
    df['Flat_Diluted Earnings Per Share'] = df['Net Profit'] / df['Avg. Diluted Shares Outstanding'] * 1000
    df['Flat_Basic PE'] = df['Share Price'] / df['Flat_Basic Earnings Per Share']
    df['Flat_Common PE'] = df['Share Price'] / df['Flat_Common Earnings Per Share']
    df['Flat_Diluted PE'] = df['Share Price'] / df['Flat_Diluted Earnings Per Share']

    # Average share prices for last 30 days
    df['Flat_SPQA'] = df['Share Price'].rolling(90, min_periods=1).mean()
    df['Flat_SPMA'] = df['Share Price'].rolling(30, min_periods=1).mean()

    # Momentum on SPMA
    try:
        df['Flat_SPMA Mom 1M'] = talib.MOM(np.array(df['Flat_SPMA']), 30)
        df['Flat_SPMA Mom 2M'] = talib.MOM(np.array(df['Flat_SPMA']), 60)
        df['Flat_SPMA Mom 3M'] = talib.MOM(np.array(df['Flat_SPMA']), 90)
        df['Flat_SPMA Mom 6M'] = talib.MOM(np.array(df['Flat_SPMA']), 180)
        df['Flat_SPMA Mom 9M'] = talib.MOM(np.array(df['Flat_SPMA']), 270)
        df['Flat_SPMA Mom 12M'] = talib.MOM(np.array(df['Flat_SPMA']), 360)
    except Exception as e:
        raise Exception(str(e))

    # Momentum on SPQA
    try:
        df['Flat_SPQA Mom 1Q'] = talib.MOM(np.array(df['Flat_SPQA']), 90)
        df['Flat_SPQA Mom 2Q'] = talib.MOM(np.array(df['Flat_SPQA']), 180)
        df['Flat_SPQA Mom 3Q'] = talib.MOM(np.array(df['Flat_SPQA']), 270)
        df['Flat_SPQA Mom 4Q'] = talib.MOM(np.array(df['Flat_SPQA']), 360)
        df['Flat_SPQA Mom 5Q'] = talib.MOM(np.array(df['Flat_SPQA']), 450)
        df['Flat_SPQA Mom 6Q'] = talib.MOM(np.array(df['Flat_SPQA']), 540)
        df['Flat_SPQA Mom 8Q'] = talib.MOM(np.array(df['Flat_SPQA']), 720)
        df['Flat_SPQA Mom 12Q'] = talib.MOM(np.array(df['Flat_SPQA']), 1080)
    except Exception as e:
        raise Exception(str(e))

    # Remove rows where Revenues is null (squash to quarterly)
    # df = df[df[flatten_by].notnull()]
    # Keep rows with X or more values (eg, Date, Ticker, CSO,BSO,DSO, Price, MarketCap, Revenue...)

    # tmp = df.loc[:, ~df.columns.str.startswith('Flat_')]
    tmp = df.loc[:, 'Revenues':'Operating Income / EV']
    tmp = tmp.dropna(thresh=5)
    return df.merge(tmp)


def flatten_by_ticker(df):

    df = df.query('Ticker == "A" | Ticker == "FLWS"')
    df = df.groupby('Ticker').apply(by_ticker)
    df.reset_index(drop=True, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

if __name__ == "__main__":

    print("Starting ...")
    # df = simfin().extract().df()
    df = df.query('Ticker == "A" | Ticker == "FLWS"')
    look = flatten_by_ticker(df)


