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


def process_by_ticker(df):
    df = df.groupby('Ticker').apply(by_ticker)
    df.reset_index(drop=True, inplace=True)
    df = df.fillna(0)

    return [df, proc]

if __name__ == "__main__":
    # df = simfin().extract().df()
    # df = df.query('Ticker == "A" | Ticker == "FLWS"')
    d = simfin().flatten().query(['FLWS', 'TSLA']).target().data_df



