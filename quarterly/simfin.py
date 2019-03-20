import pickle
import pandas as pd
from loguru import logger

# Set current directory
try:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
except NameError:
    import os
    os.chdir('d:/projects/quant/quarterly/data/staged/price')

# Set console display options for panda dataframes
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
pd.options.display.width = 150

# Load SimFin dataset
logger.info("Loading Simfin processed data set ...")
with open('../../source/simfin/data_process.pickle', 'rb') as handle:
    data = pickle.load(handle)

# Reset index
data = data.reset_index(drop=True)


data = data.query('Ticker == "FLWS"')


def byTicker(df):

    # Lag by 1 element (target)
    df['Share Price'] = df['Share Price'].shift(-1)

    return (df)

logger.info("Grouping data by ticker...")
d = data.groupby('Ticker').apply(byTicker)


