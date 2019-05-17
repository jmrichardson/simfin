import pandas as pd
from loguru import logger as log

# Check for missing quarters, insert null row and column
def by_ticker(df):

    ticker = str(df['Ticker'].iloc[0])
    # log.info("Processing {} ...".format(ticker))

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    start_date = df['Date'].iloc[0]
    end_date = df['Date'].iloc[-1]
    days = (end_date - start_date).days

    # missing_rows = df['Missing_Row'].tail(16).sum()

    # Require 4 years of data
    if days <= 1460:
        # log.warning("Not enough history")
        return pd.DataFrame()

    # df = df.tail(len(df)-16)

    return df


class History:
    def history(self):
        log.info("Getting history ...")
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker)
        self.data_df.reset_index(drop=True, inplace=True)
        self.history_df = self.data_df
        return self



