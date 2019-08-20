### SimFin

The nice folks at [SimFin](https://simfin.com/) provide freely available fundamental financial data which also includes daily pricing data.  The data can be downloaded in bulk but also requires a bit of work to prepare the data in table format.  This project is an expressive, python framework designed to process SimFin data into csv format for future analysis.  Specifically, it transforms the simfin bulk data set into quarterly table/csv format combined with most recent daily pricing data (including trailing momentum).

### Installation

Python requirements:

* pandas
* ta-lib
* loguru
* fancyimpute

Example Ubuntu/Linux Installation:

```buildoutcfg
sudo apt update
sudo apt install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
sudo ./configure
sudo make
sudo make install
pip install ta-lib
pip install pandas
pip install fancyimpute
pip install loguru
git clone https://github.com/jmrichardson/simfin
cd simfin
```

Example Windows installation in Anaconda virtual environment:

```buildoutcfg
conda create -n simfin python=3.6
conda activate simfin
conda install -y -c masdeseiscaracteres ta-lib
pip install pandas
pip install fancyimpute
pip install loguru
git clone https://github.com/jmrichardson/simfin
cd simfin
```

### Usage

First download the zipped SimFin [bulk dataset](https://simfin.com/data/access/download).  Be sure to choose "Stock prices + Fundamentals (Detailed) Dataset" as well as "wide" format with "semilcolon" delimeter before downloading.  Uncompress and move the SimFin csv file to the "simfin/data" directory:

```buildoutcfg
simfin/simfin/data/output-semicolon-wide.csv
```

An example script is provided "run.py".  Change directory to the parent simfin folder and execute the "run.py" script:
```buildoutcfg
$ python run.py
```

By default, the "run.py" script will extract the simfin dataset, flatten it into quarters with respect to daily data, remove outliers, add missing quarterly rows, remove stocks with less than 4 years history, and save the result to "data/simfin.csv":
```buildoutcfg
from simfin import *
SimFin().extract().flatten().outliers().missing_rows().history().csv()
```

### Features

Extract [SimFin bulk](https://simfin.com/data/access/api) data.   The bulk download simfin csv file requires [extraction](https://github.com/SimFin/bd-extractor) into a SimFinDataset which is enumerated into a sparse pandas dataframe.  Note, since the extract method is time consuming, the result is cached in the data folder to be used on the next invocation (remove the data/extract.pkl file to reset):

```buildoutcfg
extract()
```

Flatten the dataframe into quarterly observations while preserving the daily closing price for quarter end. The flatten() method also calculates the previous average monthly
and quarterly close as well as several momentum features for the previous year.  Note, since the flatten method is time consuming, the result is cached in the data folder to be used on the next invocation (remove the data/flatten.pkl file to reset)

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


