import pandas as pd
import numpy as np
from loguru import logger
import random
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
import os, sys

# Set current directory and initialize
try:
    os.chdir(os.path.dirname(__file__))
except:
    # Needed for working with pycharm interactive console
    script =  'd:/projects/quant/data/quarterly/simfin'
    os.chdir(script)
    sys.path.extend([script])
import init

# Set console display options for panda dataframes
pd.options.display.max_rows = 50
pd.options.display.max_columns = 60
pd.options.display.width = 150

# Load dataset
logger.info("Loading simfin dataset ...")
simfin = pd.read_pickle("data/quarterly_features.pickle")

# Temporarily make simfin dataset smaller for testing
# simfin = simfin.query('Ticker == "A" | Ticker == "AAMC" | Ticker == "FLWS"')
simfin = simfin.query('Ticker == "FLWS"')

# Process data by ticker
def by_ticker(df):

    ticker = str(df['Ticker'].iloc[0])
    logger.info("Processing " + ticker + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Add calculated features
    # dfts = pd.DataFrame(df['Date']).join(impute(df.loc[:, 'Share Price':]))
    # tf = extract_features(dfts, column_id = 'Date')
    # df = df.join(tf)


    extracted_features = extract_features(timeseries, column_id="id", column_sort="time")


    # df = df.dropna(axis = 0, thresh = 15, subset = df.columns.to_list()[3:])

    return df

logger.info("Grouping SimFin data by ticker...")
data = simfin.groupby('Ticker').apply(by_ticker)




df = simfin.groupby('Ticker').apply(by_ticker)

df = df.reset_index(drop=True)
df = pd.concat([df, pd.Series(np.nan)], ignore_index=True).drop(0, axis=1)
df_roll, y = make_forecasting_frame(df['Revenues'], kind="price", max_timeshift=16, rolling_direction=1)
X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value", impute_function=impute)
X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value")

df = df.join(X)


# tf = extract_features(dfts, column_id='Date')
# df = df.join(tf)








df = data.dropna(axis = 0)
y = pd.Series(df['Target_y Value'].values, index = df['Ticker'])
df = df.loc[:, 'Date':'Diluted PE Mom 1Q']
df = df.set_index(df['Ticker'])
X = extract_features(df, column_id = "Ticker", column_sort = "Date")
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





new = extract_relevant_features(data, y, column_id = 'Ticker', column_sort = 'Date')






# Save dataset output
data.to_csv('data' + str(random.randint(1, 100000)) + '.csv', encoding = 'utf-8', index = False)

logger.info("Saving dataframe ...")
with open('data_process.pickle', 'wb') as handle:
    pickle.dump(data, handle)




