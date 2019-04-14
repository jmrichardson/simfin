from fastai.tabular import *
from loguru import logger as log

proc={}

# Check for missing quarters, insert null row and column
def by_ticker(df):

    global proc

    ticker = str(df['Ticker'].iloc[0])
    log.info("Processing {} ...".format(ticker))

    df = df.sort_values(by='Date')

    # Remove rows with empty target
    df = df[df[[c for c in df if c.startswith('Target')]].notnull().iloc[:, 0]]

    index = df.loc[:, ['Date', 'Ticker']]
    df = df.drop(['Date', 'Ticker'], axis=1)
    X = df.filter(regex=r'^(?!Target).*$')
    y = df.filter(regex=r'^Target$')

    cols = X.columns

    missing = FillMissing(cat_names=[], cont_names=cols)
    missing(X)

    normalize = Normalize(cat_names=[], cont_names=cols)
    normalize(X)

    proc[ticker] = {"missing": missing, "normalize": normalize}

    X = X.astype(np.float64)

    df = pd.concat([index, X, y], axis=1)

    return df


class Process:
    def process(self):
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker)
        self.data_df.reset_index(drop=True, inplace=True)
        self.data_df = self.data_df.fillna(0)
        self.proc = proc
        return self


