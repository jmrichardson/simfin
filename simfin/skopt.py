from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from hyperband import HyperbandSearchCV
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier
from scipy.stats import randint, uniform
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from loguru import logger as log

from skopt.space import Real, Integer

catboost_params_space = [Real(1e-7, 1, prior='log-uniform', name='learning_rate'),
                Integer(2, 10, name='max_depth'),
                Real(0.5, 1.0, name='subsample'),
                Real(0.5, 1.0, name='colsample_bylevel'),
                Integer(1, 10, name='gradient_iterations'),
                Real(1.0, 16.0, name='scale_pos_weight'),
                Real(0.0, 1.0, name='bagging_temperature'),
                Integer(1, 20, name='random_strength'),
                Integer(2, 25, name='one_hot_max_size'),
                Real(1.0, 100, name='reg_lambda')]

# log-uniform: understand as search over p = exp(x) by varying x
search = BayesSearchCV(
    CatBoostClassifier(),
    {
        'learning_rate': Real(1e-7, 1, prior='log-uniform'),
        'max_depth': Integer(2, 14),
    },
)

log.info("Start")
search.fit(X_train, y_train)
log.info("Done")

