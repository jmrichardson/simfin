import pickle
import talib
import pandas as pd
import numpy as np
from loguru import logger

# Set current directory
try:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
except NameError:
    import os
    os.chdir('./data/fundamental/simfin')

# Set console display options for panda dataframes
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.width = 150

# Load SimFin dataset
logger.info("Loading Simfin dataset ...")
with open('simfin_dataset.pickle', 'rb') as handle:
    simfin = pickle.load(handle)

cols = len(simfin.columns)
logger.info("Dataset columns: " + str(cols))
rows = len(simfin)
logger.info("Dataset rows: " + str(rows))

# Dropping duplicates
logger.info("Dropping duplicate rows ... ")
simfin = simfin.drop_duplicates(subset=['Date', 'Ticker'])
rows = len(simfin)
logger.info("Dataset rows: " + str(rows))

# Remove rows with invalid ticker symbol
logger.info("Dropping invalid ticker rows ... ")
simfin = simfin[simfin['Ticker'].str.contains('^[A-Za-z]+$')]
rows = len(simfin)
logger.info("Dataset rows: " + str(rows))

# Temporarily make simfin dataset smaller for testing
# simfin = simfin.query('Ticker == "A" | Ticker == "AAMC"')


# Process data by ticker
def byTicker(df):

    ticker = str(df['Ticker'].iloc[0])
    logger.info("Processing " + ticker + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Fill Share Price NAs with last known value
    df['Share Price'] = df['Share Price'].ffill()

    # Add average monthly share price column
    df.insert(3, column='Share Price Monthly Average', value=df.groupby(pd.Grouper(key='Date', freq='M'))['Share Price'].transform('mean'))

    # Last not null value in previous x days
    df.iloc[:, 3:] = df.iloc[:, 3:].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)

    # Reduce to monthly by getting last info/row per month
    df = df.groupby(pd.Grouper(key='Date', freq='M')).tail(1)

    # Add technical indicators
    df.insert(4, column='MOM1MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 1))
    df.insert(5, column='MOM2MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 2))
    df.insert(6, column='MOM3MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 3))
    df.insert(7, column='MOM6MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 6))
    df.insert(8, column='MOM9MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 9))
    df.insert(9, column='MOM12MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 12))

    # Remove rows with too many null values
    df = df.dropna(axis=0, thresh=15, subset=df.columns.to_list()[3:])

    return df

logger.info("Grouping SimFin data by ticker...")
data = simfin.groupby('Ticker').apply(byTicker)

# Save dataset output
data.to_csv('data.csv', encoding='utf-8', index=False)
