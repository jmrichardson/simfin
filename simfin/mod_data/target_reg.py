import pandas as pd
import numpy as np
from loguru import logger as log

def by_ticker(df, field, lag, thresh):

    ticker = str(df['Ticker'].iloc[0])
    log.info(f"Adding target field {field} for {ticker}...")

    field_name = 'Target'
    df.loc[:, field_name] = df[field].shift(lag)
    df.loc[:, field_name] = (df[field_name] - df[field]) / df[field]

    # Remove rows where target is nan
    df = df[pd.notnull(df[field_name])]

    df[field_name] = df['Target'].where(df['Target'] > 0, other=0)
    df[field_name] = df['Target'].where(df['Target'] <= 0, other=1)

    # df.loc[:, field_name] = df[field_name].pct_change(fill_method=None)
    if thresh:
        df[field_name] = df[field_name].map(lambda x: 1 if x >= thresh else (np.nan if np.isnan(x) else 0))
    return df


def target_by_ticker(df, field, lag, thresh):
    df = df.groupby('Ticker').apply(by_ticker, field, lag, thresh)
    df.reset_index(drop=True, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


if __name__ == "__main__":
    # df = simfin().extract().df()
    # df = df.query('Ticker == "A" | Ticker == "FLWS"')
    look = target_by_ticker(df, 'SPMA', -1, .05)

