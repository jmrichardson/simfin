import os
import sys
import pandas as pd
import re
import importlib
from loguru import logger as log

# Get working directory
try:
    path = os.path.dirname(os.path.realpath(__file__))
except:
    path = 'd:/projects/quant/quant/data/simfin'

# Set path to allow for imports
os.chdir(path)
home = re.sub(r"(.*quant).*", r"\1", path)
sys.path.extend([home, path])

# Set logging parameters
log.add("logs/quarterly_tsfresh.log")

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
simfin = pd.read_pickle("data/quarterly_raw.pickle")

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
data.to_pickle("data/quarterly_features.pickle")


