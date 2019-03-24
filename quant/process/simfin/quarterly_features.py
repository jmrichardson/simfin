import os
import sys
import pandas as pd
import re
import importlib
from loguru import logger as log

# Set current directory
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

# Load SimFin dataset
log.info("Loading Simfin dataset ...")
simfin = pd.read_pickle("tmp/quarterly_raw.pickle")

# Temporarily make simfin dataset smaller for testing
# simfin = simfin.query('Ticker == "A" | Ticker == "AAMC" | Ticker == "FLWS"')
simfin = simfin.query('Ticker == "FLWS"')


# Process data by ticker
def byTicker(df):
    log.info("Processing " + str(df['Ticker'].iloc[0]) + "...")

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    # Trailing twelve month
    df = process.TTM(df, config.simfin_features, 4, 370)

    # Momentum
    df = process.mom(df, config.simfin_features, 6)
    return df

log.info("Grouping simFin data by ticker ...")
data = simfin.groupby('Ticker').apply(byTicker)
data.reset_index(drop=True, inplace=True)

log.info("Saving data ...")
data.to_pickle("tmp/quarterly_features.pickle")


