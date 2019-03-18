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

# Key features
features = ['Revenues', 'COGS', 'SG&A', 'R&D', 'EBIT', 'EBITDA', 'Net Profit',
            'Cash, Cash Equivalents & Short Term Investments', 'Cash & Cash Equivalents',
            'Receivables', 'Current Assets', 'Net PP&E', 'Total Assets', 'Short term debt', 'Accounts Payable',
            'Current Liabilities', 'Long Term Debt', 'Total Liabilities', 'Share Capital', 'Total Equity',
            'Free Cash Flow', 'Gross Margin', 'Operating Margin', 'Net Profit Margin', 'Return on Equity',
            'Return on Assets', 'Current Ratio', 'Liabilities to Equity Ratio', 'Debt to Assets Ratio',
            'EV / EBITDA', 'EV / Sales', 'Book to Market', 'Operating Income / EV', 'Enterprise Value',
            'Basic Earnings Per Share', 'Common Earnings Per Share', 'Diluted Earnings Per Share',
            'Basic PE', 'Common PE', 'Diluted PE']

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
# simfin = simfin.query('Ticker == "FLWS"')
simfin = simfin.query('Ticker == "TSLA"')


# Calculate trailing twelve months
def TTM(df, features):
    def lastYearSum(series):
        # Must have 4 quarters
        if len(series) <= 3:
            return np.nan
        # Must be within a one year date range
        else:
            firstDate = df['Date'][series.head(1).index.item()]
            lastDate = df['Date'][series.tail(1).index.item()]
            if (lastDate - firstDate).days > 370:
                return np.nan
            else:
                return series.sum()
    for feature in features:
        index = df.columns.get_loc(feature)
        df.insert(index+1, column=feature + ' TTM', value=df[feature].rolling(4, min_periods=1).apply(lastYearSum, raw=False))
    return df


def mom(df, features):
    for feature in features:
        index = df.columns.get_loc(feature)
        try:
            df.insert(index+1, column=feature + ' Mom 6Q', value=talib.MOM(np.array(df[feature]), 6))
            df.insert(index+1, column=feature + ' Mom 5Q', value=talib.MOM(np.array(df[feature]), 5))
            df.insert(index+1, column=feature + ' Mom 4Q', value=talib.MOM(np.array(df[feature]), 4))
            df.insert(index+1, column=feature + ' Mom 3Q', value=talib.MOM(np.array(df[feature]), 3))
            df.insert(index+1, column=feature + ' Mom 2Q', value=talib.MOM(np.array(df[feature]), 2))
            df.insert(index+1, column=feature + ' Mom 1Q', value=talib.MOM(np.array(df[feature]), 1))
        except:
            pass
    return df


# Process data by ticker
def byTicker(df):

    ticker = str(df['Ticker'].iloc[0])
    logger.info("Processing " + ticker + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Fill Share Price NAs with last known value
    df['Share Price'] = df['Share Price'].ffill()

    # Average share prices for last 30 days
    df.insert(3, column='SPMA', value=df['Share Price'].rolling(30, min_periods=1).mean())

    # Momentum on SPMA
    try:
        df.insert(4, column='SPMA Mom 1M', value=talib.MOM(np.array(df['SPMA']), 30))
        df.insert(5, column='SPMA Mom 2M', value=talib.MOM(np.array(df['SPMA']), 60))
        df.insert(6, column='SPMA Mom 3M', value=talib.MOM(np.array(df['SPMA']), 90))
        df.insert(7, column='SPMA Mom 6M', value=talib.MOM(np.array(df['SPMA']), 180))
        df.insert(8, column='SPMA Mom 9M', value=talib.MOM(np.array(df['SPMA']), 270))
        df.insert(9, column='SPMA Mom 12M', value=talib.MOM(np.array(df['SPMA']), 360))
    except:
        pass

    # Get Last known value (these fields are reported at different times in the quarter by simfin)
    # This will get the last value within 90 days to be used by rows with published quarter data
    # df.iloc[:, 3:] = df.iloc[:, 3:].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Common Shares Outstanding'] = df['Common Shares Outstanding'].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Avg. Basic Shares Outstanding'] = df['Avg. Basic Shares Outstanding'].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Avg. Diluted Shares Outstanding'] = df['Avg. Diluted Shares Outstanding'].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)

    # Remove rows where feature is null (removes many rows)
    df = df[df['Revenues'].notnull()]
    df = df[df['Net Profit'].notnull()]

    # Add ratios
    df['Basic Earnings Per Share'] = df['Net Profit'] / df['Avg. Basic Shares Outstanding'] * 1000
    df['Common Earnings Per Share'] = df['Net Profit'] / df['Common Shares Outstanding'] * 1000
    df['Diluted Earnings Per Share'] = df['Net Profit'] / df['Avg. Diluted Shares Outstanding'] * 1000
    df['Basic PE'] = df['Share Price'] / df['Basic Earnings Per Share']
    df['Common PE'] = df['Share Price'] / df['Common Earnings Per Share']
    df['Diluted PE'] = df['Share Price'] / df['Diluted Earnings Per Share']

    # Remove rows with too many null values
    df = df.dropna(axis=0, thresh=15, subset=df.columns.to_list()[3:])

    # Trailing twelve month on key features
    # df = TTM(df, features)

    # Momentum on key features
    # df = mom(df, features)

    return df

logger.info("Grouping SimFin data by ticker...")
data = simfin.groupby('Ticker').apply(byTicker)

# Save dataset output
data.to_csv('data' + str(random.randint(1, 100000)) + '.csv', encoding='utf-8', index=False)



