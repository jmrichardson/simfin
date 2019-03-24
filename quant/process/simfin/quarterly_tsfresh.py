import pandas as pd
from loguru import logger
import random
from tsfresh import extract_features, select_features, extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
import os
import re
import sys
import importlib

try:
    path = os.path.dirname(__file__)
    os.chdir(path)
except:
    # Python shell
    path = 'd:/projects/quant/quant/process/simfin'
    os.chdir(path)

# Import quant module(s)
home = re.sub(r"(.*quant).*", r"\1", path)
sys.path.extend([home, path])
# from config import *
import config
out = importlib.reload(config)
import process
out = importlib.reload(process)
import common
out = importlib.reload(common)

# Load dataset
logger.info("Loading simfin dataset ...")
simfin = pd.read_pickle("tmp/quarterly_features.pickle")

# Temporarily make simfin dataset smaller for testing
simfin = simfin.query('Ticker == "A" | Ticker == "AAMC" | Ticker == "FLWS"')
# simfin = simfin.query('Ticker == "FLWS"')

# Process data by ticker
def by_ticker(df):

    logger.info("Processing " + str(df['Ticker'].iloc[0]) + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    df.reset_index(drop=True, inplace=True)

    for feature in config.simfin_features:
        print(feature)
        df_roll, y = make_forecasting_frame(df['COGS'], kind="price", max_timeshift=16, rolling_direction=1)
        X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value",
                             impute_function=impute, disable_progressbar=True, show_warnings=False)
        X = X.add_prefix(feature + '_')
        X = pd.DataFrame().append(pd.Series(), ignore_index=True).append(X, ignore_index=True)
        df = df.join(X)

    # df = df.reindex(index)
    return df

logger.info("Grouping SimFin data by ticker...")
data = simfin.groupby('Ticker').apply(by_ticker)
data.reset_index(drop=True, inplace=True)




df = simfin.groupby('Ticker').apply(by_ticker)

index = df.index
for feature in ['Revenues','COGS','EBIT']:
    print(feature)
    df.reset_index(drop=True, inplace=True)
    df_roll, y = make_forecasting_frame(df['COGS'], kind="price", max_timeshift=16, rolling_direction=1)
    X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value",
                         impute_function=impute, disable_progressbar=True, show_warnings=False)
    X = X.add_prefix(feature + '_')
    X = pd.DataFrame().append(pd.Series(), ignore_index=True).append(X, ignore_index=True)
    df = df.join(X)


df = df.reindex(index)



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




