import catboost as cb
import catboost.datasets as cbd
import catboost.utils as cbu
import numpy as np
import pandas as pd
import hyperopt
import sys
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import accuracy_score


# Groups need to be grouped together
# X_train = X_train.sort_values(by=['Ticker', 'Date'], ascending=[True, True]).reset_index(drop=True)
# groups = X_train['Ticker']
# X_train = X_train.drop(['Date', 'Ticker'], axis=1)
# train_pool = Pool(X_train, y_train, group_id=groups)
# train_pool = Pool(X_train, y_train, group_id=groups)

params = {
    "iterations": 100,
    'learning_rate': 0.1,
    'eval_metric': 'Accuracy',
    "depth": 2,
    "loss_function": "Logloss",
    'use_best_model': True,
}


model = CatBoostClassifier(
    # custom_loss='Accuracy',
    # custom_metric="F1",
    eval_metric="F1",
    # logging_level='Silent',
    # eval_metric="AUC",
)

model.fit(
    X_train, y_train,
    # eval_set=(X_val, y_val),
    logging_level='Verbose',
)

