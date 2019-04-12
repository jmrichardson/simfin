import pandas as pd
import numpy as np
from loguru import logger as log

def by_ticker(df, field, lag):

    ticker = str(df['Ticker'].iloc[0])
    log.info(f"Adding target field {field} for {ticker}...")

    # Create target field
    field_name = 'Target'

    # Lag
    df.loc[:, field_name] = df[field].shift(lag)

    # Pct Diff
    df.loc[:, field_name] = (df[field_name] - df[field]) / df[field]

    # Remove rows where target is nan
    # df = df[pd.notnull(df[field_name])]

    return df


def target_reg_by_ticker(df, field, lag):
    df = df.groupby('Ticker').apply(by_ticker, field, lag)
    df.reset_index(drop=True, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


if __name__ == "__main__":
    # df = simfin().extract().df()
    # df = df.query('Ticker == "A" | Ticker == "FLWS"')
    look = target_by_ticker(df, 'SPMA', -1, .05)

