import talib
import pandas as pd
import numpy as np
from loguru import logger
import os, sys

# Set current directory and initialize
try:
    os.chdir(os.path.dirname(__file__))
except:
    # Needed for working with pycharm interactive console
    script = 'd:/projects/quant/data/quarterly/simfin'
    os.chdir(script)
    sys.path.extend([script])
import init


# Set console display options for panda dataframes
pd.options.display.max_rows = 100
pd.options.display.max_columns = 60
pd.options.display.width = 150

# Key features
features = ['Revenues', 'COGS', 'SG&A', 'R&D', 'EBIT', 'EBITDA', 'Net Profit',
            'Cash & Cash Equivalents', 'Receivables', 'Current Assets',  'Total Assets', 'Short term debt', 'Accounts Payable',
            'Current Liabilities', 'Long Term Debt', 'Total Liabilities', 'Share Capital', 'Total Equity',
            'Free Cash Flow', 'Gross Margin', 'Operating Margin', 'Net Profit Margin', 'Return on Equity',
            'Return on Assets', 'Current Ratio', 'Liabilities to Equity Ratio', 'Debt to Assets Ratio',
            'EV / EBITDA', 'EV / Sales', 'Book to Market', 'Operating Income / EV', 'Enterprise Value',
            'Basic Earnings Per Share', 'Common Earnings Per Share', 'Diluted Earnings Per Share',
            'Basic PE', 'Common PE', 'Diluted PE']

# Load SimFin dataset
logger.info("Loading Simfin dataset ...")
simfin = pd.read_pickle("data/quarterly_raw.pickle")

# Temporarily make simfin dataset smaller for testing
# simfin = simfin.query('Ticker == "A" | Ticker == "AAMC" | Ticker == "FLWS"')
# simfin = simfin.query('Ticker == "FLWS"')


# Momentum
def mom(df):
    for feature in features:
        index = df.columns.get_loc(feature)
        try:
            df[feature + ' Mom 6Q'] = talib.MOM(np.array(df[feature]), 6)
            df[feature + ' Mom 5Q'] = talib.MOM(np.array(df[feature]), 5)
            df[feature + ' Mom 4Q'] = talib.MOM(np.array(df[feature]), 4)
            df[feature + ' Mom 3Q'] = talib.MOM(np.array(df[feature]), 3)
            df[feature + ' Mom 2Q'] = talib.MOM(np.array(df[feature]), 2)
            df[feature + ' Mom 1Q'] = talib.MOM(np.array(df[feature]), 1)
        except:
            pass
    return df


# Calculate trailing twelve months
def TTM(df):
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
        # index = df.columns.get_loc(feature)
        # df.insert(index+1, column=feature + ' TTM', value=df[feature].rolling(4, min_periods=1).apply(lastYearSum, raw=False))
        df[feature + ' TTM'] = df[feature].rolling(4, min_periods=1).apply(lastYearSum, raw=False)
    return df


# Process data by ticker
def byTicker(df):

    ticker = str(df['Ticker'].iloc[0])
    logger.info("Processing " + ticker + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Trailing twelve month on key features
    df = TTM(df)

    # Momentum on key features
    df = mom(df)

    return df

logger.info("Grouping simFin data by ticker ...")
data = simfin.groupby('Ticker').apply(byTicker)

logger.info("Saving data ...")
data.to_pickle("data/quarterly_features.pickle")


