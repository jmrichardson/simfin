import pandas as pd
from loguru import logger as log
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from catboost import CatBoostClassifier
import numpy as np
import sys
from importlib import reload, import_module
from sklearn.feature_selection import VarianceThreshold


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
             target.Target,
             catboost_target.CatboostTarget,
             predict_features.PredictFeatures,
             history.History,
             tsf.TSF,
             model.Model,
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

    def sample(self, frac=0.3):
        log.info(f"Getting {frac} random sample ...")
        self.data_df = self.data_df.sample(frac=frac)
        return self

    def load(self, path=os.path.join('tmp', 'simfin')):
        if os.path.exists(path):
            log.info(f"Loading cache from {path} ...")
            return pickle.load(open(path, "rb"))

    def split(self, validation=True, test_size=.2):

        log.info("Splitting data set ...")

        self.data_df = self.data_df.sort_values(by='Date').reset_index(drop=True)

        # Get all independent features
        self.X = self.data_df.loc[:, self.data_df.columns != 'Target']

        # Get dependent feature
        self.y = self.data_df.loc[:,'Target'].values.ravel()

        # Split without shuffle (Better to split with respect to date)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, shuffle=False, random_state=1)

        if validation:
            self.X_train_split, self.X_val_split, self.y_train_split, self.y_val_split = train_test_split(
                self.X_train, self.y_train, test_size=test_size, shuffle=False , random_state=1)

        # Need groups together for catboost (may not ever use but just in case)
        # self.X_train = self.X_train.sort_values(by=['Ticker', 'Date'], ascending=[True, True]).reset_index(drop=True)
        self.groups = self.X_train['Ticker']
        self.groups_split = self.X_train_split['Ticker']

        self.X = self.X.drop(['Date', 'Ticker'], axis=1)
        self.X_train = self.X_train.drop(['Date', 'Ticker'], axis=1)
        self.X = self.X.astype(float)
        self.X_train = self.X_train.astype(float)
        self.X_train_split = self.X_train_split.drop(['Date', 'Ticker'], axis=1)
        self.X_train_split = self.X_train_split.astype(float)
        self.X_val_split = self.X_val_split.drop(['Date', 'Ticker'], axis=1)
        self.y_train_split = self.y_train_split.astype(float)
        self.X_test = self.X_test.drop(['Date', 'Ticker'], axis=1)
        self.X_test = self.X_test.astype(float)
        return self

    def important_features(self, thresh=0, include_regex="", exclude_regex=""):

        self.temp_df = self.data_df

        if 'Target' not in self.data_df.columns:
            log.warning("No target.. Adding default target to model ...")
            self = self.target()

        if len(include_regex) > 0:
            self.data_df = pd.concat([self.temp_df.loc[:, ['Date', 'Ticker']],
                                      self.data_df.filter(regex="(" + include_regex + ")"),
                                      self.data_df['Target']], axis=1)

        if len(exclude_regex) > 0:
            self.data_df = self.data_df.filter(regex="^((?!(" + exclude_regex + ")).)*$")

        self = self.split()
        log.info("Generating feature importance model ....")
        model = CatBoostClassifier(od_type='Iter', od_wait=50, verbose=0)
        model.fit(self.X_train_split, self.y_train_split, eval_set=(self.X_val_split, self.y_val_split))
        df = pd.DataFrame(model.feature_importances_).T
        df.columns = self.X_train_split.columns.values.tolist()
        df = df.T.sort_values(by=0, ascending=False)
        df.columns = ['feature']

        df = df.loc[df.iloc[:, 0] > thresh]
        self.feature_importance = df
        self.data_df = self.temp_df
        return self

    def select_features(self, thresh=0):

        self = self.important_features(thresh=thresh)
        columns = self.feature_importance.index
        names = ['Date', 'Ticker']
        names = names + self.X_train_split.loc[1, columns].index.values.tolist()
        names.append('Target')
        self.data_df = self.data_df.loc[:, names]
        self = self.split()
        return self


if __name__ == "__main__":

    # Enable logging
    log_file = os.path.join('logs', "simfin_{time:YYYY-MM-DD_HH-mm-ss}.log")
    lid = log.add(log_file, retention=5)

    # Remove log
    log.remove(lid)

