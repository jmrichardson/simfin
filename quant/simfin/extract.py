import re
import pandas as pd
from loguru import logger as log
import os
import sys

# Set current working directory (except for interactive shell)
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = 'd:/projects/quant/quant/data/simfin'

# Extend path for local imports
os.chdir(cwd)
rootPath = re.sub(r"(.*quant).*", r"\1", cwd)
sys.path.extend([cwd, rootPath])

from extractor import *

# Set logging parameters
log.add("logs/simfin.log")

class simfin:

    def __init__(self, csv='data/output-semicolon-wide.csv'):
        self.csv = csv
        self.sparseDF = pd.DataFrame()
        self.flatDF = pd.DataFrame()

    def load(self):
        # Load dataset into DF
        log.info("Converting dataset into data frame.  Be patient ...")
        df = pd.DataFrame()
        for i, company in enumerate(dataset.companies):
            df = pd.DataFrame()
            df['Date'] = dataset.timePeriods
            df['Ticker'] = company.ticker
            for i, indicator in enumerate(company.data):
                df[indicator.name] = indicator.values
            df = df.append(df)

        # Convert columns to proper format
        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop duplicates
        df.drop_duplicates(subset=['Date', 'Ticker'], keep=False, inplace=True)

        # Remove rows with invalid ticker symbol
        df = df[df['Ticker'].str.contains('^[A-Za-z]+$')]

        self.sparseDF = df










# Load SimFin dataset
# Download with "Publishing Date, Quarters, Wide, Semicolon
# dataset = SimFinDataset('output-semicolon-wide.csv', 'semicolon', "2014-12-31", "2016-12-31")
log.info("Loading simfin csv dataset.  Be patient ...")
dataset = SimFinDataset('data/output-semicolon-wide.csv', 'semicolon')

# Load dataset into Pandas DataFrame
log.info("Converting dataset into data frame.  Be patient ...")
simfin = pd.DataFrame()
for i, company in enumerate(dataset.companies):
    df = pd.DataFrame()
    df['Date'] = dataset.timePeriods
    df['Ticker'] = company.ticker
    for i, indicator in enumerate(company.data):
        df[indicator.name] = indicator.values
    simfin = simfin.append(df)

# Convert columns to proper format
simfin['Date'] = pd.to_datetime(simfin['Date'], format="%Y-%m-%d")
for col in  simfin.columns[2:]:
    simfin[col] = pd.to_numeric(simfin[col], errors='coerce')

# Drop duplicates
simfin.drop_duplicates(subset=['Date', 'Ticker'], keep=False, inplace=True)

# Remove rows with invalid ticker symbol
simfin = simfin[simfin['Ticker'].str.contains('^[A-Za-z]+$')]

# Load previous dataset
if os.path.isfile('data/extract.pickle'):
    log.info("Merging previous simfin data.  Be patient ...")
    origSimfin = pd.read_pickle("data/extract.pickle")
    mergedSimfin = pd.concat([origSimfin, simfin])
    simfin = mergedSimfin.drop_duplicates(subset=['Date', 'Ticker'], keep="last")

# Save DataFrame to pickle file for later use
log.info("Saving simfin dataframe ...")
simfin.to_pickle("data/extract.pickle")

