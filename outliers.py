from pyod.models import knn
from pyod.models import hbos
from decimal import Decimal


def by_ticker(df):


    # clf = hbos.HBOS()
    clf = knn.KNN()

    X = df.drop(['Date', 'Ticker'], axis=1)

    # Fix incorrect decimal placement
    def dec(field):
        median = field.median()
        mdp = str(float(median)).replace('-','').find('.')
        if mdp > 1:
            field = field.transform(lambda x: pow(10, mdp) *
                    float(("." + str(Decimal(pow(10, str(float(x))[::-1].find('.')) *
                    x)).replace(".", "")).replace(".-", "-.")) if (x < 1) & (x > -1) else x)
        return field
    for col in X.columns:
        X[col] = X[col].transform(dec)



    field = X['COGS'].values.reshape(-1, 1)
    field = X['Avg. Diluted Shares Outstanding'].values.reshape(-1, 1)
    out = clf.fit(field)
    clf.predict(field)
    clf.predict_proba(field)

    out = clf.fit(X)
    out = clf.fit(X['Net Profit'])

    clf.predict(X)
    X.shape
    b = X[clf.predict_proba(X)[:, 1] < .8]



class Outliers:
    def outliers(self):

        log.info("Remove outliers ...")

        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker)


