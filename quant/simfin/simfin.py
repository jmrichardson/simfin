import re
import pandas as pd
from loguru import logger as log
import os
import sys

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
from features import *

# Enable logging
lid = log.add("logs/simfin_{time:YYYY-MM-DD_HH-mm-ss}.log", retention=5)


class simfin:

    def __init__(self):
        self.bulk = pd.DataFrame()
        self.flat = pd.DataFrame()
        self.flatIndicators = pd.DataFrame()
        self.flatIndicatorsTsfresh = pd.DataFrame()

    def extractBulk(self, csv='data/output-semicolon-wide.csv', force=False):

        file = 'data/bulk.pickle'
        if not force and os.path.exists(file):
            if os.path.exists(file):
                log.info("Loading saved bulk data set ...")
                self.bulk = pd.read_pickle(file)
                return self

        log.info("Loading bulk data set.  Be patient ...")
        dataSet = SimFinDataset(csv)

        # Load dataSet into DF
        log.info("Converting data set into flat data frame.  Be patient ...")
        df = pd.DataFrame()
        for i, company in enumerate(dataSet.companies):
            df = pd.DataFrame()
            df['Date'] = dataSet.timePeriods
            df['Ticker'] = company.ticker
            for i, indicator in enumerate(company.data):
                df[indicator.name] = indicator.values
            df = df.append(df)

        # Convert columns to proper format
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop duplicates
        df.drop_duplicates(subset=['Date', 'Ticker'], keep=False, inplace=True)

        # Remove rows with invalid ticker symbol
        df = df[df['Ticker'].str.contains('^[A-Za-z]+$')]

        self.bulk = df
        return self

    # Flattened dataset
    def flatten(self, force=False):

        # Load previously saved
        file = 'data/flat.pickle'
        if not force and os.path.exists(file):
            log.info("Loading saved flat data set ...")
            self.flat = pd.read_pickle(file)
            return self

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
        self.flat = self.bulk.groupby('Ticker').apply(byTicker)
        return self

    # Add indicators for each feature
    def addIndicators(self, force=False, file='data/flat_indicators.pickle'):

        # Load previously saved
        if not force and os.path.exists(file):
            log.info("Loading saved indicator data set ...")
            self.flatIndicators = pd.read_pickle(file)
            return self

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

        log.info("Adding indicators per feature ...")
        self.flatIndicators = self.flat.groupby('Ticker').apply(byTicker)

        return self

    def addTsfresh(self, force=False, file='data/flat_indicators_tsfresh.pickle'):

        # Load previously saved data set
        if not force and os.path.exists(file):
            log.info("Loading saved tsfresh data set ...")
            self.flatIndicatorsTsfresh = pd.read_pickle(file)
            return self

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
        data = self.flatIndicators.groupby('Ticker').apply(by_ticker)
        data.reset_index(drop=True, inplace=True)
        self.flatIndicatorsTsfresh = data

        return self



# sf = simfin().extractBulk().flatten().addIndicators()
sf = simfin().flatten().addIndicators().addTsfresh()


os.mkdir("tmp")





