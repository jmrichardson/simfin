import pandas as pd
from loguru import logger as log
import pickle
from importlib import reload, import_module, util

# Import all modules (reload for testing purposes)
import os
from glob import glob
for module in glob(os.path.join('modules', '*.py')):
    module_name = os.path.basename(module)[:-3]
    exec(f"import {module_name}")
    exec(f"reload({module_name})")
    exec(f"from {module_name} import *")


class SimFin:

    def __init__(self):

        self.force = force
        self.tmp_dir = 'tmp'
        self.data_dir = 'data'
        self.process_list = []
        self.models = []

        self.data_df = pd.DataFrame

        self.csv_file = os.path.join(self.tmp_dir, csv_file)

        self.extract_df_file = os.path.join(self.tmp_dir, 'extract.zip')

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
        self.flatten_df = self.data_df
        self.data_df.to_pickle(self.flatten_df_file)

        return self

    # Add features (Indicators)
    def features(self):

        log.info("Add features by ticker ...")
        self.data_df = features_by_ticker(self.data_df, key_features)
        self.features_df = self.data_df
        return self

    def missing_rows(self):

        log.info("Add missing rows ...")
        self.data_df = missing_rows_by_ticker(self.data_df)
        self.missing_rows_df = self.data_df
        return self


    def tsf(self):

        log.info("Add tsfresh fields by ticker ...")
        self.data_df = tsf_by_ticker(self.data_df, key_features)
        self.tsf_df = self.data_df
        return self

    def target_class(self, field='Flat_SPQA', lag=-2, thresh=None):

        log.info("Adding classification target ...")
        self.data_df = target_class_by_ticker(self.data_df, field, lag, thresh)
        self.target_class_df = self.data_df
        return self

    def target_reg(self, field='Revenues', lag=-1):

        log.info("Adding regression target ...")
        self.data_df = target_reg_by_ticker(self.data_df, field, lag)
        self.target_reg_df = self.data_df
        return self

    def process(self):

        # Fill missing, normalize and save for tranforms for future prediction
        log.info("Pre-processing data ...")
        self.data_df, self.proc = process_by_ticker(self.data_df)
        self.process_df = self.data_df
        return self

    def history(self):

        # Fill missing, normalize and save for tranforms for future prediction
        log.info("Getting history ...")
        self.data_df = history_by_ticker(self.data_df)
        self.history_df = self.data_df
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

    def save(self, path=os.path.join('tmp', 'simfin')):
        log.info(f"Saving to {path} ...")
        pickle.dump(self, open(path, "wb"))
        return self

    def load(self, path=os.path.join('tmp', 'simfin')):
        if os.path.exists(path):
            log.info(f"Loading cache from {path} ...")
            return pickle.load(open(path, "rb"))

    def predict_rf_reg(self, lag=-1, max_depth=10, max_features="sqrt", min_samples_leaf=5, n_estimators=100):
        log.info("Predicting key features ...")
        self.data_df = predict_rf_reg_feature(self.data_df, lag, max_depth, max_features, min_samples_leaf, n_estimators)
        self.predict_xgboost_df = self.data_df
        return self


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


