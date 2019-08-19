from loguru import logger as log
import pickle
import pandas as pd
from importlib import reload

# Import all modules (reload for testing purposes)
import os
from glob import glob
for module in glob(os.path.join('modules', '*.py')):
    module_name = os.path.basename(module)[:-3]
    # log.info(f"Importing module: {module_name}")
    exec(f"import {module_name}")
    exec(f"reload({module_name})")
    exec(f"from {module_name} import *")


class SimFin(flatten.Flatten,
             outliers.Outliers,
             history.History,
             impute.Impute,
             missing_rows.MissingRows,
             extract.Extract):

    def __init__(self):
        self.force = force
        self.tmp_dir = 'data'
        self.data_dir = 'data'
        self.process_list = []
        self.models = []
        self.data_df = pd.DataFrame
        self.csv_file = os.path.join(self.data_dir, csv_file)
        self.extract_df_file = os.path.join(self.tmp_dir, 'extract.pkl')
        self.flatten_df_file = os.path.join(self.tmp_dir, 'flatten.pkl')

    def csv(self, file_name='data.csv'):
        path = os.path.join('data', file_name)
        log.info("Writing csv file: {}".format(path))
        self.data_df.to_csv(path, index=False)
        return self

    def query(self, tickers):
        log.info("Filtering data set")
        self.data_df = self.data_df[self.data_df['Ticker'].isin(tickers)]
        return self

    def save(self, path=os.path.join('data', 'simfin')):
        log.info(f"Saving to {path} ...")
        pickle.dump(self, open(path, "wb"))
        return self

    def load(self, path=os.path.join('data', 'simfin')):
        if os.path.exists(path):
            log.info(f"Loading cache from {path} ...")
            return pickle.load(open(path, "rb"))

    def sample(self, frac=0.3):
        log.info(f"Getting {frac} random sample ...")
        self.data_df = self.data_df.sample(frac=frac)
        return self


if __name__ == "__main__":

    # Enable logging
    log_file = os.path.join('logs', "simfin_{time:YYYY-MM-DD_HH-mm-ss}.log")
    lid = log.add(log_file, retention=5)

    # Remove log
    log.remove(lid)

