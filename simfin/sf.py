import pandas as pd
from loguru import logger as log
import pickle
from importlib import reload

# Import helper modules - FORCE RELOAD DURING TESTING - REMOVE THIS
import config
out = reload(config)
from config import *

import mod_data.extract
out = reload(mod_data.extract)
from mod_data.extract import *

import mod_data.flatten
out = reload(mod_data.flatten)
from mod_data.flatten import *

import mod_data.features
out = reload(mod_data.features)
from mod_data.features import *

import mod_data.tsf
out = reload(mod_data.tsf)
from mod_data.tsf import *

import mod_data.missing_rows
out = reload(mod_data.missing_rows)
from mod_data.missing_rows import *

import mod_data.target
out = reload(mod_data.target)
from mod_data.target import *

import mod_data.process
out = reload(mod_data.process)
from mod_data.process import *

import mod_data.history
out = reload(mod_data.history)
from mod_data.history import *

class SimFin:

    def __init__(self):

        self.force = force
        self.tmp_dir = 'tmp'
        self.data_dir = 'data'
        self.process_list = []

        self.data_df = pd.DataFrame

        self.csv_file = os.path.join(self.tmp_dir, csv_file)

        self.extract_df = pd.DataFrame()
        self.extract_df_file = os.path.join(self.tmp_dir, 'extract.zip')

        self.flatten_df = pd.DataFrame()
        self.flatten_df_file = os.path.join(self.tmp_dir, 'flatten.zip')


    def extract(self):

        # Load previously saved DF if exists
        if not self.force and os.path.exists(self.extract_df_file):
            if os.path.exists(self.extract_df_file):
                log.info("Loading saved extract data set ...")
                self.extract_df = pd.read_pickle(self.extract_df_file)
                self.data_df = self.extract_df
                return self

        self.data_df = extract_bulk(self.csv_file)
        self.data_df.to_pickle(self.extract_df_file)

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
        self.data_df = flatten_by_ticker(self.data_df)
        self.data_df.to_pickle(self.flatten_df_file)

        return self

    # Add features for each feature
    def features(self):

        log.info("Add features by ticker ...")
        self.data_df = features_by_ticker(self.data_df, key_features)
        return self

    def missing_rows(self):

        log.info("Add missing rows ...")
        self.data_df = missing_rows_by_ticker(self.data_df)
        return self


    def tsf(self):

        log.info("Add tsfresh fields by ticker ...")
        self.data_df = tsf_by_ticker(self.data_df, key_features)
        return self

    def target(self, field='Flat_SPQA', lag=-4, thresh=None):

        log.info("Add target ...")
        self.data_df = target_by_ticker(self.data_df, field, lag, thresh)
        return self

    def process(self):

        # Fill missing, normalize and save for tranforms for future prediction
        log.info("Pre-processing data ...")
        self.data_df, self.proc = process_by_ticker(self.data_df)
        return self

    def history(self):

        # Fill missing, normalize and save for tranforms for future prediction
        log.info("Getting history ...")
        self.data_df = history_by_ticker(self.data_df)
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

    def save(self, file='simfin'):
        path = os.path.join(self.tmp_dir, file + '.pkl')
        log.info(f"Saving to {path} ...")
        pickle.dump(self, open(path, "wb"))
        return self

    def load(self, file='simfin'):
        path = os.path.join(self.tmp_dir, file + '.pkl')
        if os.path.exists(path):
            log.info(f"Loading cache from {path} ...")
            return pickle.load(open(path, "rb"))


if __name__ == "__main__":

    # Enable logging
    log_file = os.path.join('logs', "simfin_{time:YYYY-MM-DD_HH-mm-ss}.log")
    lid = log.add(log_file, retention=5)


    # df = SimFin().flatten().query(['FLWS']).csv('flws.csv').data_df
    # df = SimFin().flatten().query(['FLWS']).data_df
    # df = SimFin().flatten().query(['TSLA']).data_df
    # df = SimFin().flatten().query(['FLWS','TSLA']).data_df
    # df = SimFin().flatten().query(['FLWS','TSLA']).missing_rows().data_df
    # df = SimFin().flatten().query(['FLWS','TSLA']).missing_rows().target().process().data_df
    # df = SimFin().flatten().target().process().data_df
    # df = SimFin().flatten().query(['FLWS','TSLA']).target().process().data_df
    sf = SimFin().flatten().target().process().save('rf')
    # SimFin = SimFin().flatten().save()
    # df = SimFin().flatten().query(['FLWS','TSLA']).target().data_df
    # df = SimFin().flatten().query(['TSLA']).data_df
    # df = SimFin().flatten().target().data_df
    # df = SimFin().flatten().query(['FLWS','TSLA']).data_df
    # new  = SimFin().flatten().query(['FLWS','TSLA']).target().process()
    # df = SimFin().flatten().query(['FLWS','TSLA']).target().process().data_df


    # Remove log
    log.remove(lid)


