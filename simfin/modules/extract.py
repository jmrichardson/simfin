"""
Step 1:  Extract simfin dataset (simfin format) into large (sparse) pandas dataframe.
"""

# Import required packages
import pandas as pd
from loguru import logger as log
from modules.extractor import *
import os


# Extract simfin dataset
def extract_bulk(csv_file):

    log.info("Loading bulk csv data set.  Be patient ...")
    dataSet = SimFinDataset(csv_file)

    # Load dataSet into pandas data frame
    log.info("Converting data set into data frame.  Be patient ...")
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
    return data[data['Ticker'].str.contains('^[A-Za-z]+$')]


# Class to extract simfin data
class Extract:

    # Extract method to extract simfin data
    def extract(self):

        # Load previously saved DF if exists
        if not self.force and os.path.exists(self.extract_df_file):
            if os.path.exists(self.extract_df_file):
                log.info("Loading previously saved extracted data set ...")
                self.extract_df = pd.read_pickle(self.extract_df_file)
                self.data_df = self.extract_df
                return self

        # Extract simfin csv file
        self.data_df = extract_bulk(self.csv_file)

        # extract_df_file is defined in sf.py where class is initiated
        # save to pickle file (default tmp/extract.zip)
        self.data_df.to_pickle(self.extract_df_file)

