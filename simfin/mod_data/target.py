import pandas as pd
import numpy as np
from loguru import logger as log

def by_ticker(df, field, lag, thresh):

    field_name = 'Target'
    df[field_name] = df[field].pct_change(fill_method=None)
    df[field_name] = df[field_name].shift(lag)
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

