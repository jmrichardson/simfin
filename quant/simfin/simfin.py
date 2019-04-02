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
import features
out = reload(features)
from features import *
import tsf
out = reload(tsf)
from tsf import *
import missing
out = reload(missing)
from missing import *
import target
out = reload(target)
from target import *




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

        self.features_df = pd.DataFrame()
        self.tsf_df = pd.DataFrame()
        self.missing_df = pd.DataFrame()

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

    # Add features for each feature
    def features(self):

        log.info("Add features by ticker ...")
        self.features_df = features_by_ticker(self.data_df, key_features)
        self.data_df = self.features_df
        return self

    def missing(self):

        log.info("Add missing rows ...")
        self.missing_df = missing_by_ticker(self.data_df)
        self.data_df = self.missing_df
        return self

    def tsf(self):

        log.info("Add tsfresh fields by ticker ...")
        self.tsf_df = tsf_by_ticker(self.data_df, key_features)
        self.data_df = self.tsf_df
        return self

    def target(self, field='Flat_SPMA', lag=1, thresh=None):

        log.info("Add target ...")
        self.target_df = target_by_ticker(self.data_df, field, lag, thresh)
        self.data_df = self.target_df
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




if __name__ == "__main__":

    # Enable logging
    log_file = os.path.join('logs', "simfin_{time:YYYY-MM-DD_HH-mm-ss}.log")
    lid = log.add(log_file, retention=5)


    # df = simfin().flatten().query(['FLWS']).csv('flws.csv').data_df
    # df = simfin().flatten().query(['FLWS','TSLA']).missing().features().tsf().csv()
    # df = simfin().flatten().query(['FLWS','TSLA']).missing().target().csv()
    df = simfin().flatten().query(['FLWS','TSLA']).missing().features().data_df



    # Remove log
    log.remove(lid)


