import pandas as pd
from loguru import logger as log
from extractor import *


def extract_bulk(csv_file):
    log.info("Loading bulk csv data set.  Be patient ...")
    dataSet = SimFinDataset(csv_file)

    # Load dataSet into Df
    log.info("Converting data set into flat data frame.  Be patient ...")
    data = pd.DataFrame()
    for i, company in enumerate(dataSet.companies):
        df = pd.DataFrame()
        df['Date'] = dataSet.timePeriods
        df['Ticker'] = company.ticker
        for i, indicator in enumerate(company.data):
            df[indicator.name] = indicator.values
        data = data.append(df)

    # Convert columns to proper format
    data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
    for col in data.columns[2:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop duplicates
    data.drop_duplicates(subset=['Date', 'Ticker'], keep=False, inplace=True)

    # Remove rows with invalid ticker symbol
    data = data[data['Ticker'].str.contains('^[A-Za-z]+$')]

    return data
