import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold, cross_validate
from loguru import logger as log
from config import *


class PredictFeatures:

    def predict_features(self):

        self = self.process()
        self = self.split()

        for feature in key_features:

            log.info(f"Predicting feature ...")

            self = self.target(field=feature)

            scale = (len(self.y_train) - sum(self.y_train)) / len(self.y_train)
            params = {
                'learning_rate': .025,
                'task_type': 'GPU',
                'scale_pos_weight': scale,
                'od_type': 'Iter',
                'od_wait': 50,
                'verbose': 1,
            }
            model = CatBoostClassifier(**params, random_state=123)
            model.fit(
                self.X_train_split, self.y_train_split,
                eval_set=(self.X_val_split, self.y_val_split)
            )

            params = {**params, **{'n_estimators': model.best_iteration_}}

            model = CatBoostClassifier(**params, random_state=123)
            model.fit(
                self.X_train, self.y_train,
            )

            self.data_df['Catboost_Class_' + feature] = model.predict(self.X)

        return self


