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
import config
from config import *
out = importlib.reload(config)

# Load dataset
logger.info("Loading simfin dataset ...")
simfin = pd.read_pickle("tmp/quarterly_features.pickle")

# Temporarily make simfin dataset smaller for testing
# simfin = simfin.query('Ticker == "A" | Ticker == "AAMC" | Ticker == "FLWS"')
# simfin = simfin.query('Ticker == "FLWS"')


# Process data by ticker
def by_ticker(df):

    logger.info("Processing " + str(df['Ticker'].iloc[0]) + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    df.reset_index(drop=True, inplace=True)

    for feature in simfin_features:
        print(feature)
        df_roll, y = make_forecasting_frame(df['COGS'], kind="price", max_timeshift=16, rolling_direction=1)
        X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value",
                             impute_function=impute, disable_progressbar=True, show_warnings=False)
        X = X.add_prefix('tsfresh_' + feature + '_')
        X = pd.DataFrame().append(pd.Series(), ignore_index=True).append(X, ignore_index=True)
        df = df.join(X)

    # df = df.reindex(index)
    return df

logger.info("Grouping SimFin data by ticker...")
data = simfin.groupby('Ticker').apply(by_ticker)
data.reset_index(drop=True, inplace=True)

log.info("Saving data ...")
data.to_pickle("tmp/quarterly_tsfresh.pickle")


### Select relevant features

# Per ticker, pct change, shift by -1 then remove last nan row
y = simfin.groupby('Ticker')['SPMA'].apply(lambda x: x.pct_change().shift(periods=-1)[:-1])
y = y.reset_index(drop=True)
# Per ticker, remove last row because there is nan for y
X = data.groupby('Ticker').apply(lambda df: df[:-1])
X = X.loc[:, data.columns.str.startswith('tsfresh_')]
X = X.reset_index(drop=True)
X = impute(X)
new = select_features(X, y)



