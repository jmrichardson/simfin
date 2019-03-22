import talib
import pandas as pd
import numpy as np
from loguru import logger
import random
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
import os

# Set current directory
try:
    os.chdir(os.path.dirname(__file__))
except:
    os.chdir('d:/projects/quant/data/source/simfin')

# Set console display options for panda dataframes
pd.options.display.max_rows = 100
pd.options.display.max_columns = 60
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
simfin = pd.read_pickle("extract.pickle")

# Temporarily make simfin dataset smaller for testing
# simfin = simfin.query('Ticker == "A" | Ticker == "AAMC" | Ticker == "FLWS"')
simfin = simfin.query('Ticker == "FLWS"')
# simfin = simfin.query('Ticker == "TSLA"')
# simfin = simfin.query('Ticker == "ABR"')
# simfin = simfin.head(500000)


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


# Calculate trailing 24 months
def T24M(df):
    def yearSum(series):
        # Must have 8 quarters
        if len(series) <= 7:
            return np.nan
        # Must be within a one year date range
        else:
            series = series.head(4)
            firstDate = df['Date'][series.head(1).index.item()]
            lastDate = df['Date'][series.tail(1).index.item()]
            if (lastDate - firstDate).days > 370:
                return np.nan
            else:
                return series.sum()
    for feature in features:
        # index = df.columns.get_loc(feature)
        # df.insert(index+1, column=feature + ' T24M', value=df[feature].rolling(8, min_periods=1).apply(yearSum, raw=False))
        df[feature + ' T24M'] = df[feature].rolling(8, min_periods=1).apply(yearSum, raw=False)
    return df


# lag prediction targets
def target(df, features):
    for feature in features:
        df['Target_y ' + feature] = (df[feature].shift(-1) - df[feature])/df[feature]
    return df[:-1]


# Process data by ticker
def byTicker(df):

    ticker = str(df['Ticker'].iloc[0])
    logger.info("Processing " + ticker + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Fill Share Price NAs with last known value
    df['Share Price'] = df['Share Price'].ffill()

    # Get Last known value (these fields are reported at different times in the quarter by simfin)
    # This will get the last value within 90 days to be used by rows with published quarter data
    # df.iloc[:, 3:] = df.iloc[:, 3:].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Common Shares Outstanding'] = df['Common Shares Outstanding'].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Avg. Basic Shares Outstanding'] = df['Avg. Basic Shares Outstanding'].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
    df['Avg. Diluted Shares Outstanding'] = df['Avg. Diluted Shares Outstanding'].rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)

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

    # Need at least 3 quarters
    if len(df.index) <= 3:
        logger.warning(" - Not enough history")
        return None

    return df

    # Add calculated features
    # dfts = pd.DataFrame(df['Date']).join(impute(df.loc[:, 'Share Price':]))
    # tf = extract_features(dfts, column_id='Date')
    # df = df.join(tf)



    # Trailing twelve month on key features
    # df = T24M(df)
    # df = TTM(df)

    # Momentum on key features
    # df = mom(df)

    # Add lagged target for features
    # df = target(df, ['SPQA'] + features)

    # Add value target if percent gain is greater than x percent
    # df['Target_y Value'] = np.where(df['Target_y SPQA'] >= .05, 1, 0)

    # Scale all fundamental features by last Market Cap (not by row to show relative change in values)
    # marketCap = df['Market Capitalisation'].tail(1).item()
    # df.loc[:, 'Common Shares Outstanding':'Diluted PE Mom 1Q'] = df.loc[:, 'Common Shares Outstanding':'Diluted PE Mom 1Q'] / marketCap

    # Remove rows with too many null values
    # df = df.dropna(axis=0, thresh=15, subset=df.columns.to_list()[3:])

    return df

logger.info("Grouping SimFin data by ticker...")
data = simfin.groupby('Ticker').apply(byTicker)




df = simfin.groupby('Ticker').apply(byTicker)

df = df.reset_index(drop=True)
dfShift, y = make_forecasting_frame(df['EBITDA'], kind="price", max_timeshift=16, rolling_direction=1)
X = extract_features(dfShift, column_id="id", column_sort="time", column_value="value", impute_function=impute)
X = pd.concat([pd.Series(np.nan), X], ignore_index=True)
df = df.join(X)


# tf = extract_features(dfts, column_id='Date')
# df = df.join(tf)








df = data.dropna(axis=0)
y = pd.Series(df['Target_y Value'].values, index=df['Ticker'])
df = df.loc[:,'Date':'Diluted PE Mom 1Q']
df = df.set_index(df['Ticker'])
X = extract_features(df, column_id="Ticker", column_sort="Date")
X = impute(X)
new = select_features(X, y)

j = X.loc[:, X.apply(pd.Series.nunique) != 1]



# df = data.loc[:, 'Share Price':'Diluted PE Mom 1Q']


# df = pd.DataFrame(data['Date']).join(impute(data.loc[:, 'Share Price':'Diluted PE Mom 1Q']))
# df = data.loc[:, 'Share Price':'Diluted PE Mom 1Q']
# df = impute(df)
# # df = df.set_index(data['Ticker'])
# df = df.reset_index(drop=True)
# y = data['Target_y Value']
# # y = pd.Series(data['Target_y Value'].values, index=data['Ticker'])
# y = y.reset_index(drop=True)



new = select_features(df, y)
#

y = data['Target_y Value']
y.reindex(data['Date'])
index = y.index.get_level_values(0).to_series().tolist()
y.set_index(index)





new = extract_relevant_features(data, y, column_id='Ticker', column_sort='Date')






# Save dataset output
data.to_csv('data' + str(random.randint(1, 100000)) + '.csv', encoding='utf-8', index=False)

logger.info("Saving dataframe ...")
with open('data_process.pickle', 'wb') as handle:
    pickle.dump(data, handle)




