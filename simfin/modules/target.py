import pandas as pd
import numpy as np
from loguru import logger as log

def by_ticker(df, field, type, lag, thresh):

    # ticker = str(df['Ticker'].iloc[0])
    # log.info(f"Adding target field {field} for {ticker}...")

    # Create target field
    field_name = 'Target'

    # Lag
    df.loc[:, field_name] = df[field].shift(lag)

    # Pct Diff
    df.loc[:, field_name] = (df[field_name] - df[field]) / df[field]

    # Remove rows where target is nan
    # df = df[pd.notnull(df[field_name])]

    # If thresh, then 1 if > thresh, else 0
    if type == 'class':
        if thresh is None:
            df[field_name] = df[field_name].map(lambda x: 1 if x > 0 else (np.nan if np.isnan(x) else 0))
        else:
            df[field_name] = df[field_name].map(lambda x: 1 if x >= thresh else (np.nan if np.isnan(x) else 0))


    # Class 0 or 1:  < 0 -> 0, > 1 -> 1
    # df[field_name] = df['Target'].where(df['Target'] > 0, other=0)
    # df[field_name] = df['Target'].where(df['Target'] <= 0, other=1)

    return df


class Target:
    def target(self, field='Flat_SPQA', type='class', lag=-1, thresh=None):
        log.info("Adding target ...")
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker, field, type, lag, thresh)
        self.data_df.reset_index(drop=True, inplace=True)
        self.data_df = self.data_df.replace([np.inf, -np.inf], np.nan)

        # Remove null target rows and sort by date
        self.data_df = self.data_df[pd.notnull(self.data_df['Target'])].sort_values(by='Date')

        return self


