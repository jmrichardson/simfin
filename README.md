### SimFin

The nice folks at [SimFin](https://simfin.com/) provide freely available fundamental financial data which also includes daily pricing data.  The information can be downloaded in bulk but also requires a bit of work to prepare the data for analysis.  This project is an expressive python framework designed to process SimFin data into a flat CSV file for future analysis.

### Installation

Requirements

* pandas
* ta-lib
* loguru
* fancyimpute

Example installation in conda virtual environment:

```buildoutcfg
conda create -n simfin python=3.6
conda activate simfin
conda install -y -c masdeseiscaracteres ta-lib
pip install pandas
pip install fancyimpute
pip install loguru
```

### Usage

First download the zipped SimFin [bulk dataset](https://simfin.com/data/access/download) and uncompress/move the csv file to the "simfin/data" directory.  Be sure to choose "Stock prices + Fundamentals (Detailed) Dataset" as well as "wide" format with "semilcolon" delimeter when downloading the dataset. The downloaded SimFin dataset should have the following path:

```buildoutcfg
simfin/simfin/data/output-semicolon-wide.csv
```

An example run script is provided "run.py".  Change directory to the parent simfin folder and execute the "run.py" script (If you installed the required python packages in a virtual environment, be sure to activate it):
```buildoutcfg
(simfin)$ python run.py
```

By default, the "run.py" script will extract the simfin dataset, flatten it into quarters with respect to daily data, remove outliers, add missing quarterly rows, remove stocks with less than 4 years history, and save the result to "data/data.csv":
```buildoutcfg
from simfin import *
simfin = SimFin().extract().flatten().outliers().missing_rows().history().csv()
```

### Features

Extract [SimFin bulk](https://simfin.com/data/access/api) data.   The bulk download simfin csv file requires [extraction](https://github.com/SimFin/bd-extractor) into a SimFinDataset. This dataset is enumerated into a sparse pandas dataframe (daily combined with quarterly data) using the extract() method.  Note, since this process is time consuming, the result is cached in the tmp folder to be used on the next invocation (remove the tmp/extract.zip file if starting over or new SimFin dataset):

```buildoutcfg
extract()
```

Flatten the dataframe into quarterly observations while preserving daily closing price history. The flatten() method calculates close monthly and quarterly averages and creates new features for 1, 2, 3, 6, 9 and 12 previous months and 1, 2, 3, 4, 5, 6, 8 and 12 quarters respectively. In addition to SimFin's ratios, a handful of other feature ratios are calculated. Note, since this process is time consuming, the result is cached in the tmp folder to be used on the next invocation (remove the tmp/flatten.zip file if starting over or new SimFin dataset)

```buildoutcfg
flatten()
```

Add empty rows for missing quarters. If applying temporal analysis, it may make sense to add empty rows for missing quarters.  A new field is automatically added indicating the row contains missing data.

```buildoutcfg
missing_rows()
```

Remove outliers using inter-quartile range (IQR).
```buildoutcfg
outliers()
```

Impute missing data points using KNN.

```buildoutcfg
impute()
```

Remove stock data without enough history.  Some stocks in the simfin data set do not provide a significant amount of historical data.  Stocks with less than 4 years of data are removed.

```buildoutcfg
history()
```

Filter dataset by ticker.  

```buildoutcfg
query(["AAPL", "MSFT"])
```

Randomly sample percentage of dataset.

```buildoutcfg
sample(frac=.2)
```

### New Repo Coming ...

I will be creating a new repo with features to enrich datasets such as the SimFin with additional information for analysis


