from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFECV
from xgboost import XGBClassifier

scale = (len(y_train) - sum(y_train)) / len(y_train)
classifier = XGBClassifier(
    n_estimators=80,
    learning_rate=0.08,
    max_depth=30,
    subsample=0.7,
    colsample_bytree=0.55,
    min_child_weight=3,
    gamma=0.5,
    reg_alpha=1e-05,
    scale_pos_weight=scale,
    seed=1,
    silent=True
)

cv = GroupKFold(n_splits=10)
selector = RFECV(estimator=classifier, step=1, cv=cv, scoring='balanced_accuracy')
selector.fit(X_train, y_train)


