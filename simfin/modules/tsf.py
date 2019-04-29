import pandas as pd
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from loguru import logger as log
import os
from config import *
from multiprocessing import freeze_support


# Process data by ticker
def by_ticker(df):

    global key_features

    # Log ticker
    ticker = str(df['Ticker'].iloc[0])
    log.info("Ticker: " + ticker)

    rows = len(df)
    if rows < 2:
        log.warning("Not enough history(rows): {} actual, 2 required".format(rows))
        return pd.DataFrame

    # Sort dataframe by date
    df = df.sort_values(by='Date')
    df.reset_index(drop=True, inplace=True)

    # For each key feature, add tsfresh calculation columns
    tmp_df = pd.DataFrame
    for feature in key_features:
        log.info("  Feature: " + feature)

        # Must have at least 2 rows of data
        if df[feature].count() <= 1:
            log.warning("  Feature count <= 1: " + feature)
            continue

        # Make rolling forecast frame from past 16 Quarters
        df_roll, y = make_forecasting_frame(df[feature], kind="price", max_timeshift=16,
                                            rolling_direction=1)
        # Calculate
        X = extract_features(df_roll, column_id="id", column_sort="time", column_value="value",
                             disable_progressbar=True, show_warnings=False)
        # Add prefix to each column
        X = X.add_prefix('tsfresh_' + feature + '_')
        # empty row to beginning of X to make sure same number of rows as original df (first row dropped from calc)
        X = pd.DataFrame().append(pd.Series(), ignore_index=True).append(X, ignore_index=True)
        if tmp_df.empty:
            tmp_df = X
        else:
            tmp_df = tmp_df.join(X)

    # Join all calculated feature columns to original df
    df = df.join(tmp_df)
    return df


class TSF:
    def tsf(self):

        # Need to insert missing rows because rolling history
        self = self.missing_rows()

        log.info("Add tsfresh fields by ticker ...")
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker).reset_index(drop=True, inplace=False)
        return self


'''
 log.info("Grouping SimFin data by ticker...")
    data = simfin.groupby('Ticker').apply(by_ticker)
    data.reset_index(drop=True, inplace=True)

    log.info("Saving data ...")
    data.to_pickle("data/quarterly_features_tsfresh.pickle")

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
'''

