import pandas as pd
from loguru import logger as log
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from importlib import reload, import_module

# Import all modules (reload for testing purposes)
import os
from glob import glob
for module in glob(os.path.join('modules', '*.py')):
    module_name = os.path.basename(module)[:-3]
    exec(f"import {module_name}")
    exec(f"reload({module_name})")
    exec(f"from {module_name} import *")


class SimFin(flatten.Flatten,
             features.Features,
             process.Process,
             predict_rf.PredictRF,
             target.Target,
             history.History,
             tsf.TSF,
             missing_rows.MissingRows,
             extract.Extract):

    def __init__(self):
        self.force = force
        self.tmp_dir = 'tmp'
        self.data_dir = 'data'
        self.process_list = []
        self.models = []
        self.data_df = pd.DataFrame
        self.csv_file = os.path.join(self.data_dir, csv_file)
        self.extract_df_file = os.path.join(self.tmp_dir, 'extract.zip')
        self.flatten_df_file = os.path.join(self.tmp_dir, 'flatten.zip')

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

    def split(self, test_size=.2):
        df = self.data_df[pd.notnull(self.data_df['Target'])]
        X = df.filter(regex=r'^(?!Target).*$')
        y = df.filter(regex=r'^Target$')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=1)
        return self


if __name__ == "__main__":

    # Enable logging
    log_file = os.path.join('logs', "simfin_{time:YYYY-MM-DD_HH-mm-ss}.log")
    lid = log.add(log_file, retention=5)

    # Remove log
    log.remove(lid)

