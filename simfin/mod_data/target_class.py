import pandas as pd
import numpy as np
from loguru import logger as log

def by_ticker(df, field, lag, thresh):

    ticker = str(df['Ticker'].iloc[0])
    log.info(f"Adding target field {field} for {ticker}...")

    # Create target field
    field_name = 'Target'

    # Lag
    df.loc[:, field_name] = df[field].shift(lag)

    # Pct Diff
    df.loc[:, field_name] = (df[field_name] - df[field]) / df[field]

    # Remove rows where target is nan
    df = df[pd.notnull(df[field_name])]


    # If thresh, then 1 if > thresh, else 0
    if thresh:
        df[field_name] = df[field_name].map(lambda x: 1 if x >= thresh else (np.nan if np.isnan(x) else 0))
    else:
        df[field_name] = df[field_name].map(lambda x: 1 if x > 0 else (np.nan if np.isnan(x) else 0))

    # Class 0 or 1:  < 0 -> 0, > 1 -> 1
    # df[field_name] = df['Target'].where(df['Target'] > 0, other=0)
    # df[field_name] = df['Target'].where(df['Target'] <= 0, other=1)

    return df


def target_class_by_ticker(df, field, lag, thresh):
    df = df.groupby('Ticker').apply(by_ticker, field, lag, thresh)
    df.reset_index(drop=True, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


if __name__ == "__main__":
    # df = simfin().extract().df()
    # df = df.query('Ticker == "A" | Ticker == "FLWS"')
    look = target_by_ticker(df, 'SPMA', -1, .05)

