from fastai.tabular import *
from loguru import logger as log

# Check for missing quarters, insert null row and column
def by_ticker(self):

    ticker = str(df['Ticker'].iloc[0])
    log.info("Processing {} ...".format(ticker))

    # Remove rows with empty target
    df = df[df[[c for c in df if c.startswith('Target_')]].notnull().iloc[:, 0]]

    df = df.sort_values(by='Date').drop(['Date', 'Ticker'], axis=1)
    X = df.filter(regex=r'^(?!Target_).*$')
    y = df.filter(regex=r'Target_.*')

    cols = X.columns

    missing = FillMissing(cat_names=[], cont_names=cols)
    missing(X)

    normalize = Normalize(cat_names=[], cont_names=cols)
    normalize(X)

    df = pd.concat([X, y], axis=1)

    return df

def process_by_ticker(self):
    # df, missing, normalize = df.groupby('Ticker').apply(by_ticker)
    a = self.data_df.groupby('Ticker').apply(by_ticker, self)
    return self


if __name__ == "__main__":
    # df = simfin().extract().df()
    # df = df.query('Ticker == "A" | Ticker == "FLWS"')
    df = simfin().flatten().query(['FLWS', 'TSLA']).target().data_df
    ret, out = df.groupby('Ticker').apply(by_ticker)



