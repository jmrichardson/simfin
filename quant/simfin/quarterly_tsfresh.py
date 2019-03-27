import pandas as pd
from loguru import logger as log
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
import os
import re
import sys
import importlib

# Get working directory
try:
    path = os.path.dirname(os.path.realpath(__file__))
except:
    path = 'd:/projects/quant/quant/data/simfin'

# Set path to allow for imports
os.chdir(path)
home = re.sub(r"(.*quant).*", r"\1", path)
sys.path.extend([home, path])

# Import required modules
import config
from config import *
out = importlib.reload(config)


def main():

    # Set logging parameters
    lid = log.add("logs/quarterly_tsfresh_{time:YYYY-MM-DD_HH-mm-ss}.log", retention=5)

    # Load dataset
    log.info("Loading simfin dataset ...")
    simfin = pd.read_pickle("data/quarterly_features.pickle")

    # simfin = simfin.query('Ticker == "AIR"')


    # Process data by ticker
    def by_ticker(df):

        ticker = str(df['Ticker'].iloc[0])
        store = "tmp/" + ticker + ".pickle"
        log.info("Ticker: " + ticker)

        if os.path.isfile(store):
            df = pd.read_pickle(store)
            return df

        # Sort dataframe by date
        df = df.sort_values(by='Date')

        df.reset_index(drop=True, inplace=True)

        for feature in simfin_features:
            log.info("  Feature: " + feature)
            if df[feature].count() <= 1:
                log.warning("  Feature count <= 1: " + feature)
                continue

            df_roll, y = make_forecasting_frame(df[feature], kind="price", max_timeshift=16, rolling_direction=1)
            # X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value",
                                 # impute_function=impute, disable_progressbar=True, show_warnings=False)
            X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value",
                                 disable_progressbar=True, show_warnings=False)
            X = X.add_prefix('tsfresh_' + feature + '_')
            X = pd.DataFrame().append(pd.Series(), ignore_index=True).append(X, ignore_index=True)
            df = df.join(X)

        log.info("writing " + store)
        df.to_pickle(store)

        return df

    log.info("Grouping SimFin data by ticker...")
    data = simfin.groupby('Ticker').apply(by_ticker)
    data.reset_index(drop=True, inplace=True)

    log.info("Saving data ...")
    data.to_pickle("data/quarterly_tsfresh.pickle")

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

    log.info("Saving data ...")
    new.to_pickle("data/quarterly_tsfresh_new.pickle")

    # Remove log
    log.remove(lid)


if __name__ == '__main__':
    main()


