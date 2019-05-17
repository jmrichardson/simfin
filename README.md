# UNDER CONSTRUCTION - ALPHA STATUS!

### SimFin Predict  

The nice folks at [SimFin](https://simfin.com/) provide freely available fundamental financial data which also includes daily pricing data.  The information can be downloaded in bulk but also requires a bit of work to prepare the data for analysis.  Simfin Predict is an expressive python framework designed to easily process SimFin information for analysis and to predict future price movement. SimFin Predict combines many machine and deep learning techniques to maximize predictive power.

### Features

Extract and flatten [SimFin bulk](https://simfin.com/data/access/api) data
* Bulk download csv file requires [extraction](https://github.com/SimFin/bd-extractor) into a SimFinDataset. This dataset is enumerated into a sparse pandas dataframe (daily combined with quarterly data) using the extract() method.  
* The extracted dataframe can then be flattened into quarterly observations while preserving daily closing price history. The flatten() method calculates close monthly and quarterly averages and creates new features for 1, 2, 3, 6, 9 and 12 previous months and 1, 2, 3, 4, 5, 6, 8 and 12 quarters respectively. 
* In addition to SimFin's ratios, a handful of other feature ratios are calculated.
```buildoutcfg
simfin = SimFin().extract().flatten()
```


Automatic feature engineering.  Using the powerful tool [featuretools](https://www.featuretools.com/), new features are created from transform primitives.
* Multi-step feature engineering rather than brute force of all possible combinations of transform primitives (ie: subtraction, addition, division).  Each step involves generating new features using a single primitive, then dimensionality is reduced using a ML model.  This avoids the significant overhead of generating potentially useless features and processing time.
* Divide the feature universe into pricing and fundamental categories due to the characteristic differences. The final step combines both buckets using the division primitive (ratio).
```buildoutcfg
simfin = simfin.engineer()
```


Add informative features such as date information and technical indicators with respect to each ticker.  
* For each ticker, calculate technical indicators such as trailing twelve months (TTM) and momentum (MOM).  Todo: add more indicators
* Date features added using [fastai datepart](https://docs.fast.ai/tabular.transform.html)
```buildoutcfg
simfin = simfin.features()
```


Add time series characteristics for each key feature using [tsfresh](https://tsfresh.readthedocs.io/en/latest/text/introduction.html).    The approach is to provide a rolling historical window to provide time components for each observation.  Thus, allowing traditional machine learning algorithms to recognize patterns in sequential data.  
* For each ticker, create a rolling 16 quarter observation window of calculated features.  
* Some quarter observations are missing from the SimFin dataset. Therefore, empty rows are inserted for missing observations to ensure the integrity of the rolling window.
* Unfortunately, tsfresh produces informative warning messages that cannot be suppressed at the moment which will flood the console.
* A significant amount of features are calculated requiring time (be patient) and resources ([list of calculated features](https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html)). 
```buildoutcfg
simfin = simfin.tsf()
```
    

Add predicted key features. The smart folks at [Euclidean Technologies](https://www.euclidean.com/) published a [paper](https://arxiv.org/pdf/1711.04837.pdf) demonstrating value in predicting future fundamentals (features).
*  Regression prediction made on all important features.
```buildoutcfg
simfin = simfin.predict_features()
```


Add Temporal Convolutional Network classification and regression features.  Thanks to [Keras TCN](https://github.com/philipperemy/keras-tcn), we generate a deep learning model to forecast price movement.
* Because observations contain patterns across quarters, 
```buildoutcfg
simfin = simfin.tcn()
```
    





### TODO

* Add macro economic data (todo)
* Add options data (todo)
* Ensemble machine and deep learning predictors
* Custom independent targets
* Backtest portfolio strategy
