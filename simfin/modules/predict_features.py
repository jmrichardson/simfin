import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from loguru import logger as log
from config import *

if GPU:
    task_type = 'GPU'
else:
    task_type = 'CPU'

class PredictFeatures:

    def predict_features(self):

        for feature in key_features:

            self = self.target(field=feature, type='regression')
            self = self.process()
            self = self.split()

            # df = simfin.data_df

            log.info(f"Predicting {feature} regression ...")
            params = {
                'learning_rate': .025,
                'task_type': task_type,
                'od_type': 'Iter',
                'od_wait': 100,
                'verbose': 0,
            }
            model = CatBoostRegressor(**params, random_state=123)
            model.fit(
                self.X_train_split, self.y_train_split,
                eval_set=(self.X_val_split, self.y_val_split)
            )
            self.data_df['Catboost_Reg_' + feature] = model.predict(self.X)

            # I don't think we should refit on entire training set
            # params = {**params, **{'n_estimators': model.best_iteration_}}
            # model = CatBoostClassifier(**params, random_state=123)
            # model.fit(
            # self.X_train, self.y_train,
            # )

            self = self.target(field=feature)
            self = self.process()
            self = self.split()

            log.info(f"Predicting {feature} classification probability ...")
            scale = (len(self.y_train) - sum(self.y_train)) / len(self.y_train)
            params = {**params, **{'scale_pos_weight': scale}}

            model = CatBoostClassifier(**params, random_state=123)
            model.fit(
                self.X_train_split, self.y_train_split,
                eval_set=(self.X_val_split, self.y_val_split)
            )
            self.data_df['Catboost_Class_' + feature] = model.predict_proba(self.X)[:, 1]

        return self


