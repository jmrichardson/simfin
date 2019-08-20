import pandas as pd
from loguru import logger as log
from config import *

# Check for missing quarters, insert null row and column
def by_ticker(df):

    ticker = str(df['Ticker'].iloc[0])

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    start_date = df['Date'].iloc[0]
    end_date = df['Date'].iloc[-1]
    days = (end_date - start_date).days

    # Require minimum number of quarter history
    if days <= 90 * min_quarters:
        # log.warning("Not enough history")
        return pd.DataFrame()

    # Must also require >X rows of non missing data
    if 'Missing_Row' in df.columns:
        total_rows = len(df)
        missing_rows = df['Missing_Row'].sum()
        quarters = total_rows - missing_rows
        if quarters < min_quarters:
            return pd.DataFrame()
    return df


class History:
    def history(self):
        log.info("Removing stocks without enough history ...")
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker)
        self.data_df.reset_index(drop=True, inplace=True)
        return self



