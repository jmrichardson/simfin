import pickle
import talib
import pandas as pd
import numpy as np
from loguru import logger
import random

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
simfin = simfin.query('Ticker == "FLWS"')


indicators = ['Revenues', 'COGS']
df (TTM):
df['Revenues'].rolling(12, min_periods=1).sum()


# Process data by ticker
def byTicker(df):

    ticker = str(df['Ticker'].iloc[0])
    logger.info("Processing " + ticker + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Fill Share Price NAs with last known value
    df['Share Price'] = df['Share Price'].ffill()

    # Average share prices for last 30 days
    df.insert(3, column='Share Price Monthly Average', value=df['Share Price'].rolling(30, min_periods=1).mean())
    try:
        df.insert(4, column='MOM1MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 30))
        df.insert(5, column='MOM2MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 60))
        df.insert(6, column='MOM3MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 90))
        df.insert(7, column='MOM6MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 180))
        df.insert(8, column='MOM9MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 270))
        df.insert(9, column='MOM12MA', value=talib.MOM(np.array(df['Share Price Monthly Average']), 360))
    except:
        pass

    # Get Last known value (these fields are reported at different times in the quarter by simfin)
    # This will get the last value within 90 days to be used by rows with published quarter data
    # df.iloc[:, 3:] = df.iloc[:, 3:].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Common Shares Outstanding'] = df['Common Shares Outstanding'].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Avg. Basic Shares Outstanding'] = df['Avg. Basic Shares Outstanding'].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Avg. Diluted Shares Outstanding'] = df['Avg. Diluted Shares Outstanding'].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)

    # Remove rows where column val is Null (Only rows with quarterly report published)
    df = df[df['Revenues'].notnull()]

    # Remove rows with too many null values
    df = df.dropna(axis=0, thresh=15, subset=df.columns.to_list()[3:])

    # Trailing values
    df['Revenues TTM'] = df['Revenues'].rolling(12, min_periods=1).sum()

    return df

logger.info("Grouping SimFin data by ticker...")
data = simfin.groupby('Ticker').apply(byTicker)

# Save dataset output
data.to_csv('data' + str(random.randint(1, 100000)) + '.csv', encoding='utf-8', index=False)



