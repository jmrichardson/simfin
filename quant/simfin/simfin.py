import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute, make_forecasting_frame
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


# Enable logging
log_file = os.path.join(log_dir, "simfin_{time:YYYY-MM-DD_HH-mm-ss}.log")
lid = log.add(log_file, retention=5)


class simfin:

    def __init__(self, force=False):
        self.force = force
        self.data_dir = data_dir
        self.tmp_dir = tmp_dir
        self.min_quarters = min_quarters
        self.flatten_by = flatten_by

        self.csv_file = os.path.join(self.data_dir, csv_file)

        self.extract_df = pd.DataFrame()
        self.extract_df_file = os.path.join(self.data_dir, 'extract.zip')

        self.flatten_df = pd.DataFrame()
        self.flatten_df_file = os.path.join(self.data_dir, 'flatten.zip')

        self.indicators_df = pd.DataFrame()
        self.indicators_df_file = os.path.join(self.data_dir, 'indicators.zip')

        self.tsfresh_df = pd.DataFrame()
        self.tsfresh_df_file = os.path.join(self.data_dir, 'tsfresh.zip')

    def extract(self):

        # Load previously saved DF if exists
        if not self.force and os.path.exists(self.extract_df_file):
            if os.path.exists(self.extract_df_file):
                log.info("Loading saved extract data set ...")
                self.extract_df = pd.read_pickle(self.extract_df_file)
                return self

        self.extract_df = extract_bulk(self.csv_file)
        self.extract_df.to_pickle(self.extract_df_file)

        return self

    # Flatten extracted bulk simfin dataset into quarterly
    def flatten(self):

        # Load previously saved DF if exists
        if not self.force and os.path.exists(self.flatten_df_file):
            log.info("Loading saved flattened data set ...")
            self.flatten_df = pd.read_pickle(self.flatten_df_file)
            return self

        # If empty bulk data, load previously saved or throw error
        if self.extract_df.empty:
            if os.path.exists(self.extract_df_file):
                log.info("Loading saved extract bulk data set ...")
                self.extract_df = pd.read_pickle(self.extract_df_file)
            else:
                raise Exception("No extracted bulk data set.  Run method extract()")

        log.info("Flattening SimFin data set into quarterly ...")
        self.flatten_df = flatten_by_ticker(self.extract_df, self.flatten_by)
        self.flatten_df.to_pickle(self.flatten_df_file)

        return self

    # Add indicators for each feature
    def indicators(self, *args):

        # Init from previously saved DF if exists
        if not self.force and os.path.exists(self.indicators_df_file):
            log.info("Loading saved indicator data set ...")
            self.indicators_df = pd.read_pickle(self.indicators_df_file)
            return self

        # Either user supplied df or load from self
        if self.flatten_df.empty:
            if args:
                if isinstance(args[0], pd.DataFrame):
                    self.flatten_df = args[0]
                else:
                    raise Exception('First argument must be data frame')
            elif os.path.exists(self.flatten_df_file):
                self.flatten_df = pd.read_pickle(self.flatten_df_file)
            else:
                raise Exception('Flattened quarterly simfin data set required.  Run method flatten()')

        if len(self.flatten_df) < self.min_quarters:
            raise Exception('Must have minimum quarters of history: ' + str(self.min_quarters))

        log.info("Add indicators by ticker ...")
        self.indicators_df = indicators_by_ticker(self.flatten_df, simfinFeatures)
        self.indicators_df.to_pickle(self.indicators_df_file)
        return self

    def tsfresh(self, *args):

        # Load previously saved data set
        if not self.force and os.path.exists(self.tsfresh_df_file):
            log.info("Loading saved tsfresh data set ...")
            self.tsfresh_df = pd.read_pickle(self.tsfresh_df_file)
            return self

        log.info("Add tsfresh indicators by ticker ...")
        self.tsfresh_df = tsfresh_by_ticker(self.indicators_df, self.tmp_dir)
        self.tsfresh_df.to_pickle(file)
        return self


# table = pd.read_pickle('data/table.zip')
# flws = table.query('Ticker == "FLWS"')

sf = simfin().extract().flatten()
# sf = simfin(force=True).indicators(flws)

# Remove log
log.remove(lid)


