from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
from MLFeatureSelection import sequence_selection, importance_selection, coherence_selection,tools
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import cross_val_predict
from MLFeatureSelection import sequence_selection

def lossfunction(y_pred, y_test):
    return balanced_accuracy_score(y_pred, y_test)


def validate(X, y, features, clf, lossfunction):
    global groups
    cv = GroupKFold(n_splits=5)
    y_pred = cross_val_predict(clf, X, y, cv=cv, groups=groups)
    score = lossfunction(y, y_pred)
    print(score)
    return score, clf

def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

def sq(x,y):
    return x ** 2


CrossMethod = {'+':add,
               '-':substract,
               '*':times,
               '/':divide,
               '^': sq,
               }

sf = sequence_selection.Select(Sequence = True, Random = True, Cross = False)
sf.ImportDF(X, label = 'Target')
sf.ImportLossFunction(lossfunction, direction='ascend')
sf.InitialNonTrainableFeatures(notusable)
# sf.InitialFeatures(initialfeatures)
sf.GenerateCol()
sf.clf = CatBoostClassifier(
    n_estimators=80,
    learning_rate=0.08,
    max_depth=30,
    subsample=0.7,
    colsample_bytree=0.55,
    min_child_weight=3,
    gamma=0.5,
    reg_alpha=1e-05,
    scale_pos_weight=4,
    seed=1,
    silent=True
)
sf.SetLogFile('record.log')
sf.run(validate)











def seq(df, f, notusable, estimator):
    sf = sequence_selection.Select(Sequence=True, Random=False, Cross=True)
    sf.ImportDF(df, label = 'Target')
    sf.ImportLossFunction(lossfunction, direction='descend')
    sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(notusable)
    sf.InitialFeatures(f)
    sf.GenerateCol()
    sf.clf = estimator
    sf.SetLogFile('record_seq.log')
    return sf.run(validate)


def imp(df, f, estimator):
    sf = importance_selection.Select()
    sf.ImportDF(df, label='Target')
    sf.ImportLossFunction(lossfunction, direction='descend')
    sf.InitialFeatures(f)
    sf.SelectRemoveMode(batch=1)
    sf.clf = estimator
    sf.SetLogFile('record_imp.log')
    return sf.run(validate)


def coh(df, f, estimator):
    sf = coherence_selection.Select()
    sf.ImportDF(df, label='Target')
    sf.ImportLossFunction(lossfunction, direction='descend')
    sf.InitialFeatures(f)
    sf.SelectRemoveMode(batch=1, lowerbound=0.5)
    sf.clf = estimator
    sf.SetLogFile('record_coh.log')
    return sf.run(validate)


def run():

    df = X
    notusable = ['Target'] #not trainable features

    # f = tools.readlog('record2.log',0.086342) # use readlog to read the out log (filename, required score)
    #f = ['SNR','BN'] #initial features combination

    clf = XGBClassifier(
        n_estimators=80,
        learning_rate=0.08,
        max_depth=30,
        subsample=0.7,
        colsample_bytree=0.55,
        min_child_weight=3,
        gamma=0.5,
        reg_alpha=1e-05,
        scale_pos_weight=4,
        seed=1,
        # silent=True
    )

    uf = f[:]
    print('sequence selection')
    uf = seq(df, uf, notusable, clf)

    print('importance selection')
    uf = imp(df,uf,clf)

    print('coherence selection')
    uf = coh(df,uf,clf)
    return uf


























from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFECV
from xgboost.sklearn import XGBClassifier
from boostaroota import BoostARoota

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
    # silent=True
)

cv = GroupKFold(n_splits=5)
selector = RFECV(estimator=classifier, step=1, cv=cv, scoring='balanced_accuracy')
selector.fit(X_train, y_train, groups=groups)



br = BoostARoota(metric='logloss')
br.fit(X_train, y_train)
br.keep_vars_
br.transform(X_train)
