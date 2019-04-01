import pandas as pd
from loguru import logger as log
import re
import os
import sys
from importlib import reload

# Set current working directory (except for interactive shell)
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = 'd:/projects/quant/quant/simfin'

# Extend path for local imports
os.chdir(cwd)
rootPath = re.sub(r"(.*quant).*", r"\1", cwd)
sys.path.extend([cwd, rootPath])

# Import helper modules - FORCE RELOAD DURING TESTING - REMOVE THIS
import config
out = reload(config)
from config import *
import extract
out = reload(extract)
from extract import *
import flatten
out = reload(flatten)
from flatten import *
import indicators
out = reload(indicators)
from indicators import *
import tsf
out = reload(tsf)
from tsf import *




class simfin:

    def __init__(self):
        self.force = force
        self.tmp_dir = 'tmp'
        self.data_dir = 'data'
        self.min_history = min_history

        self.data_df = pd.DataFrame

        self.csv_file = os.path.join(self.tmp_dir, csv_file)

        self.extract_df = pd.DataFrame()
        self.extract_df_file = os.path.join(self.tmp_dir, 'extract.zip')

        self.flatten_df = pd.DataFrame()
        self.flatten_df_file = os.path.join(self.tmp_dir, 'flatten.zip')

        self.indicators_df = pd.DataFrame()
        self.indicators_df_file = os.path.join(self.tmp_dir, 'indicators.zip')

        self.tsf_df = pd.DataFrame()
        self.tsf_df_file = os.path.join(self.tmp_dir, 'tsf.zip')

        self.missing_df = pd.DataFrame()
        self.missing_df_file = os.path.join(self.tmp_dir, 'missing.zip')

    def extract(self):

        # Load previously saved DF if exists
        if not self.force and os.path.exists(self.extract_df_file):
            if os.path.exists(self.extract_df_file):
                log.info("Loading saved extract data set ...")
                self.extract_df = pd.read_pickle(self.extract_df_file)
                self.data_df = self.extract_df
                return self

        self.extract_df = extract_bulk(self.csv_file)
        self.extract_df.to_pickle(self.extract_df_file)
        self.data_df = self.extract_df

        return self

    # Flatten extracted bulk simfin dataset into quarterly.
    def flatten(self):

        # Load previously saved DF if exists
        if not self.force and os.path.exists(self.flatten_df_file):
            log.info("Loading saved flattened data set ...")
            self.flatten_df = pd.read_pickle(self.flatten_df_file)
            self.data_df = self.flatten_df
            return self

        # If empty bulk data, load previously saved or throw error
        if self.data_df.empty:
            if os.path.exists(self.extract_df_file):
                log.info("Loading saved extract data set ...")
                self.extract_df = pd.read_pickle(self.extract_df_file)
                self.data_df = self.extract_df
            else:
                raise Exception("No extracted data set.  Run method extract()")

        log.info("Flattening SimFin data set into quarterly ...")
        self.flatten_df = flatten_by_ticker(self.data_df)
        self.data_df = self.flatten_df
        self.flatten_df.to_pickle(self.flatten_df_file)

        return self

    # Add indicators for each feature
    def indicators(self, *args):

        # Load previously saved indicators
        if not self.force and os.path.exists(self.indicators_df_file):
            log.info("Loading saved indicator data set ...")
            self.indicators_df = pd.read_pickle(self.indicators_df_file)
            self.data_df = self.indicators_df
            return self

        # If empty data_df, must provide in method call
        if self.data_df.empty:
            if args:
                if isinstance(args[0], pd.DataFrame):
                    self.data_df = args[0]
                else:
                    raise Exception('First argument must be data frame')
            else:
                raise Exception('Empty data frame.  Run flatten() or provide data frame')

        log.info("Add indicators by ticker ...")
        self.indicators_df = indicators_by_ticker(self.data_df, key_features)
        self.data_df = self.indicators_df
        self.indicators_df.to_pickle(self.indicators_df_file)
        return self


    def missing(self):
        # Load previously saved
        if not self.force and os.path.exists(self.missing_df_file):
            log.info("Loading saved tsfresh data set ...")
            self.missing_df = pd.read_pickle(self.missing_df_file)
            self.data_df = self.missing_df
            return self


        return self

    def tsf(self, *args):

        # Load previously saved tsf
        if not self.force and os.path.exists(self.tsf_df_file):
            log.info("Loading saved tsfresh data set ...")
            self.tsf_df = pd.read_pickle(self.tsf_df_file)
            self.data_df = self.tsf_df
            return self

        # If empty data_df, must provide in method call
        if self.data_df.empty:
            if args:
                if isinstance(args[0], pd.DataFrame):
                    self.data_df = args[0]
                else:
                    raise Exception('First argument must be data frame')
            else:
                raise Exception('Empty data frame.  Run flatten() or provide data frame')

        log.info("Add tsfresh indicators by ticker ...")
        self.tsf_df = tsf_by_ticker(self.data_df, key_features, self.min_history)
        self.data_df = self.tsf_df
        self.tsf_df.to_pickle(file)
        return self

    def target(self, field='SPMA', lag=1, thresh=None):
        return self


    def csv(self, file_name='data.csv'):
        path = os.path.join('data', file_name)
        log.info("Writing csv file: {}".format(path))
        self.data_df.to_csv(path)
        return self

    def query(self, tickers):
        log.info("Filtering data set")
        self.data_df = self.data_df[self.data_df['Ticker'].isin(tickers)]
        return self

    def df(self):
        log.info("Returning data set")
        return self.data_df





if __name__ == "__main__":

    # Enable logging
    log_file = os.path.join('logs', "simfin_{time:YYYY-MM-DD_HH-mm-ss}.log")
    lid = log.add(log_file, retention=5)

    # sf = simfin().flatten().tsf()
    # sf = simfin().extract().csv("data.csv")
    sf = simfin().flatten().csv("look.csv")
    # sf = simfin().flatten()

    # sf = simfin().indicators(flws)
    # df = simfin().flatten().query(['FLWS']).csv('flws.csv').df()
    # df = simfin().flatten().query(['ALJJ'])

    # Remove log
    log.remove(lid)


