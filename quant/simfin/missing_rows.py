from datetime import date
from loguru import logger as log
import pandas as pd
import numpy as np

# Check for missing quarters, insert null row and column
def by_ticker(df):

    ticker = str(df['Ticker'].iloc[0])
    log.info("Processing {} ...".format(ticker))

    # Sort dataframe by date
    df = df.sort_values(by='Date')

    start_date = df['Date'].iloc[0]
    for index, row in df.iterrows():
        end_date = row['Date']
        days = (end_date - start_date).days

        if days > 150:
            while True:
                new_date = start_date + pd.DateOffset(days=90)
                if new_date < end_date - pd.DateOffset(days=90):
                    new_df = pd.DataFrame([{'Date': new_date, 'Ticker': ticker}])
                    df = df.append(new_df, sort=False)
                    log.info("Inserting empty row: {}".format(new_date))
                    start_date = new_date
                else:
                    break
        start_date = end_date

    df = df.sort_values(by='Date')
    df.reset_index(drop=True, inplace=True)

    return df

def missing_rows_by_ticker(df):
    df = df.groupby('Ticker').apply(by_ticker)
    df.reset_index(drop=True, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


if __name__ == "__main__":
    # df = simfin().extract().df()
    # df = df.query('Ticker == "A" | Ticker == "FLWS"')
    df = missing_rows_by_ticker(df)



