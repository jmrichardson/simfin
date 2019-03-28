import re
import pandas as pd
import numpy as np
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
from loguru import logger as log
import os
import sys
from importlib import reload
import talib

# Set current working directory (except for interactive shell)
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = 'd:/projects/quant/quant/simfin'

# Extend path for local imports
os.chdir(cwd)
rootPath = re.sub(r"(.*quant).*", r"\1", cwd)
sys.path.extend([cwd, rootPath])

# Import helper modules
from extractor import *
import config
out = reload(config)
from config import *



# Enable logging
log_file = os.path.join(log_dir, "simfin_{time:YYYY-MM-DD_HH-mm-ss}.log")
lid = log.add(log_file, retention=5)


class simfin:

    def __init__(self, force=False):
        self.force = force
        self.data_dir = data_dir
        self.tmp_dir = tmp_dir
        self.csv_file = os.path.join(self.data_dir, csv_file)

        self.bulk_df = pd.DataFrame()
        self.bulk_df_file = os.path.join(self.data_dir, 'bulk.pickle')

        self.flat_df = pd.DataFrame()
        self.flat_df_file = os.path.join(self.data_dir, 'flat.pickle')

        self.indicators_df = pd.DataFrame()
        self.indicators_df_file = os.path.join(self.data_dir, 'indicators.pickle')

        self.tsfresh_df = pd.DataFrame()
        self.tsfresh_df_file = os.path.join(self.data_dir, 'tsfresh.pickle')

    def bulk(self):

        # Load previously saved DF if exists
        if not self.force and os.path.exists(self.bulk_df_file):
            if os.path.exists(self.bulk_df_file):
                log.info("Loading saved bulk data set ...")
                self.bulk_df = pd.read_pickle(self.bulk_df_file)
                return self

        log.info("Loading bulk data set.  Be patient ...")
        dataSet = SimFinDataset(self.csv_file)

        # Load dataSet into Df
        log.info("Converting data set into flat data frame.  Be patient ...")
        data = pd.DataFrame()
        for i, company in enumerate(dataSet.companies):
            df = pd.DataFrame()
            df['Date'] = dataSet.timePeriods
            df['Ticker'] = company.ticker
            for i, indicator in enumerate(company.data):
                df[indicator.name] = indicator.values
            data = data.append(df)

        # Convert columns to proper format
        data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
        for col in data.columns[2:]:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Drop duplicates
        data.drop_duplicates(subset=['Date', 'Ticker'], keep=False, inplace=True)

        # Remove rows with invalid ticker symbol
        data = data[data['Ticker'].str.contains('^[A-Za-z]+$')]

        self.bulk_df = data
        self.bulk_df.to_pickle(self.bulk_df_file)

        return self

    # Flattened dataset
    def flat(self):

        # Load previously saved DF if exists
        if not self.force and os.path.exists(self.flat_df_file):
            log.info("Loading saved flat data set ...")
            self.flat_df = pd.read_pickle(self.flat_df_file)
            return self

        # If empty bulk data, load previously saved or throw error
        if self.bulk_df.empty:
            if os.path.exists(self.bulk_df_file):
                log.info("Loading saved bulk data set ...")
                self.bulk_df = pd.read_pickle(self.bulk_df_file)
            else:
                log.error("No bulk data set.  Run bulk method")
                exit()

        # Process by ticker
        def byTicker(df):

            log.info("Processing " + str(df['Ticker'].iloc[0]) + "...")

            # Sort dataframe by date
            df = df.sort_values(by='Date')

            # Fill Share Price NAs with last known value
            df['Share Price'] = df['Share Price'].ffill()

            # Get Last known value (these fields are reported at different times in the quarter by simfin)
            # This will get the last value within 90 days to be used by rows with published quarter data
            df['Common Shares Outstanding'] = df['Common Shares Outstanding']\
                .rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
            df['Avg. Basic Shares Outstanding'] = df['Avg. Basic Shares Outstanding']\
                .rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)
            df['Avg. Diluted Shares Outstanding'] = df['Avg. Diluted Shares Outstanding']\
                .rolling(92, min_periods=1).apply(lambda x: x[x.last_valid_index()], raw=False)

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

            return df

        log.info("Flattening SimFin data set by ticker ...")
        self.flat_df = self.bulk_df.groupby('Ticker').apply(byTicker)
        self.flat_df.to_pickle(self.flat_df_file)

        return self

    # Add indicators for each feature
    def indicators(self):

        # Load previously saved DF if exists
        if not self.force and os.path.exists(self.indicators_df_file):
            log.info("Loading saved indicator data set ...")
            self.indicators_df = pd.read_pickle(self.indicators_df_file)
            return self

        # Add momentum indicator on features
        def mom(df, features, count):
            for feature in features:
                if df[feature].isnull().all():
                    continue
                for i in range(1, count + 1, 1):
                    try:
                        df[feature + ' Mom ' + str(i) + 'Q'] = talib.MOM(np.array(df[feature]), i)
                    except Exception as e:
                        log.warning("Momentum " + str(feature) + ": " + str(e))
            return df

        # Calculate trailing twelve months
        def TTM(df, features, roll, days):
            def lastYearSum(series):
                # Must have x previous inputs
                if len(series) < roll:
                    return np.nan
                # Must be within X days date range
                else:
                    firstDate = df['Date'][series.head(1).index.item()]
                    lastDate = df['Date'][series.tail(1).index.item()]
                    if (lastDate - firstDate).days > days:
                        return np.nan
                    else:
                        return series.sum()

            for feature in features:
                if df[feature].isnull().all():
                    continue
                try:
                    df[feature + ' TTM'] = df[feature].rolling(roll, min_periods=1).apply(lastYearSum, raw=False)
                except:
                    log.warning("Unable to add TTM for: " + feature)
            return df

        # Process data by ticker
        def byTicker(df):
            log.info("Processing " + str(df['Ticker'].iloc[0]) + "...")

            # Sort dataframe by date
            df = df.sort_values(by='Date')

            # Trailing twelve month
            df = TTM(df, simfinFeatures, 4, 370)

            # Momentum
            df = mom(df, simfinFeatures, 6)
            return df

        log.info("Adding indicators per feature ...")
        self.indicators_df = self.flat_df.groupby('Ticker').apply(byTicker)
        self.indicators_df.to_pickle(self.indicators_df_file)

        return self


    def tsfresh(self):

        # Load previously saved data set
        if not self.force and os.path.exists(self.tsfresh_df_file):
            log.info("Loading saved tsfresh data set ...")
            self.tsfresh_df = pd.read_pickle(self.tsfresh_df_file)
            return self

        # Process data by ticker
        def by_ticker(df):

            ticker = str(df['Ticker'].iloc[0])
            store = os.path.join(self.tmp_dir, ticker + ".pickle")
            log.info("Ticker: " + ticker)

            if os.path.isfile(store):
                df = pd.read_pickle(store)
                return df

            # Sort dataframe by date
            df = df.sort_values(by='Date')

            df.reset_index(drop=True, inplace=True)

            for feature in simfinFeatures:
                log.info("  Feature: " + feature)
                if df[feature].count() <= 1:
                    log.warning("  Feature count <= 1: " + feature)
                    continue

                df_roll, y = make_forecasting_frame(df[feature], kind="price", max_timeshift=16,
                                                    rolling_direction=1)
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

        log.info("Add tsfresh indicators by feature ...")
        data = self.indicators_df.groupby('Ticker').apply(by_ticker)
        data.reset_index(drop=True, inplace=True)
        self.tsfresh_df = data
        self.tsfresh_df.to_pickle(file)

        return self


sf = simfin().bulk().flat().indicators().tsfresh()


# Remove log
log.remove(lid)


