from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectFromModel

# Groups need to be grouped together
# X_train = X_train.sort_values(by=['Ticker', 'Date'], ascending=[True, True]).reset_index(drop=True)
# groups = X_train['Ticker']
# X_train = X_train.drop(['Date', 'Ticker'], axis=1)
# train_pool = Pool(X_train, y_train, group_id=groups)
# train_pool = Pool(X_train, y_train, group_id=groups)

params={'bagging_temperature': 2.0, 'depth': 8.0, 'eval_metric': 'F1', 'l2_leaf_reg': 46.0,
        'learning_rate': 0.012043652624016063, 'n_estimators': 492.0, 'random_strength': 79.0,
        'task_type': 'GPU', 'thread_count': -1, 'verbose': 1}

params={'eval_metric': 'F1', 'task_type': 'GPU', 'random_seed': 1}
params={'eval_metric': 'F1', 'n_estimators': 1133.0, 'task_type': 'GPU'}

model = CatBoostClassifier(**params)

model = CatBoostClassifier()

model.fit(
    X_train_split, y_train_split,
    eval_set=(X_val_split, y_val_split),
    logging_level='Verbose',
)

model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    logging_level='Verbose',
)


imp_features = pd.Series(model.feature_importances_)
imp_features = model.feature_importances_
thresh = 0
sel_cols = [True if x > thresh else False for x in imp_features]
X_train.loc[:, sel_cols]


# Get train and validation score
# train_score = model.best_score_['learn']['Precision']
# validation_score = model.best_score_['validation_0']['Precision']
# print(f'Train score: {train_score}')
# print(f'Validation score: {validation_score}')