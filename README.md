### SimFin Predict  

The nice folks at [SimFin](https://simfin.com/) provide freely available fundamental financial data which also includes daily pricing data.  The information can be downloaded in bulk but also requires a bit of work to prepare the data for analysis.  Simfin Predict is an expressive python framework designed to easily process SimFin information for analysis and to predict future price movement.

### Features

* Extract and flatten [SimFin bulk](https://simfin.com/data/access/api) data

    * Bulk download csv file requires [extraction](https://github.com/SimFin/bd-extractor) into a SimFinDataset. This dataset is enumerated into a sparse (daily combined with quarterly data) pandas dataframe using the extract() method.  
    * The extracted dataframe can then be flattened into quarterly observations while preserving daily closing price history. The flatten() method calculates close monthly and quarterly averages and creates new features for 1, 2, 3, 6, 9 and 12 previous months and 1, 2, 3, 4, 5, 6, 8 and 12 quarters. 
    * In addition to SimFin's ratios, a handful of other feature ratios are calculated.

```buildoutcfg
simfin = SimFin().extract().flatten()
```

* Add informative features with respect to each ticker symbol such as date information and technical indicators.  

    * For each ticker, calculate technical indicators such trailing twelve months (TTM) and momentum (MOM).  Todo: add more indicators
    * Date features added using [fastai datepart](https://docs.fast.ai/tabular.transform.html)

```buildoutcfg
simfin = simfin.features()
```




* Add technical indicators and time series characteristics
* Add macro economic data (todo)
* Add options data (todo)
* Ensemble machine and deep learning predictors
* Custom independent targets
* Backtest portfolio strategy