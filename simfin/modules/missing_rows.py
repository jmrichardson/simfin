from loguru import logger as log
import pandas as pd
import numpy as np


# Check for missing quarters, insert null row and column
def by_ticker(df):

    ticker = str(df['Ticker'].iloc[0])
    # log.info("Processing {} ...".format(ticker))

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    missing = False
    start_date = df['Date'].iloc[0]
    for index, row in df.iterrows():
        end_date = row['Date']
        days = (end_date - start_date).days

        if days > 150:
            while True:
                new_date = start_date + pd.DateOffset(days=90)
                if new_date < end_date - pd.DateOffset(days=90):
                    new_df = pd.DataFrame([{'Date': new_date, 'Ticker': ticker}])
                    new_df['Missing_Row'] = 1
                    df = df.append(new_df, sort=False)
                    # log.info(f"Inserting empty row for {ticker}: {new_date}")
                    start_date = new_date
                else:
                    break
        start_date = end_date
    df = df.sort_values(by='Date')

    return df


class MissingRows:
    def missing_rows(self):
        log.info("Add missing rows ...")
        self.data_df = self.data_df.assign(Missing_Row=0)
        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker)
        self.data_df['Missing_Row'].fillna(0, inplace=True)
        self.data_df = self.data_df.replace([np.inf, -np.inf], np.nan)
        self.data_df.reset_index(drop=True, inplace=True)
        self.missing_rows_df = self.data_df
        return self



