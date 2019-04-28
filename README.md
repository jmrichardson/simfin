### SimFin Predict  

Expressive python framework to:

* Parse and prepare [SimFin](https://simfin.com/) data
* Add new features such as technical indicators and time series characteristics
* Combine macro economic data (todo)
* Compine options data (todo)
* Custom independent targets
* Ensemble machine and deep learning predictors
* Backtest portfolio strategy

The nice folks at [SimFin](https://simfin.com/) provide freely available fundamental financial data which also includes daily pricing data.  The information can be downloaded in bulk but also requires a bit of work to prepare the data for analysis.  Simfin Predict (SP) is designed to easily process the information for analysis as well as use SF data and other sources to predict future price movement.

### Features

* Extract [SimFin bulk](https://simfin.com/data/access/api) data

    * Bulk download file in csv format requires [extraction](https://github.com/SimFin/bd-extractor) into a SimFinDataset. This dataset is enumerated into a sparse (daily combined with quarterly data) pandas dataframe.  
    * The extracted pandas dataframe can then be flattened into quarterly observations while preserving daily information using additional features(columns). 





