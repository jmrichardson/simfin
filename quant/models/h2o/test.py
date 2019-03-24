
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
plt.style.use('seaborn')
import seaborn as sns

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from sklearn.ensemble import AdaBoostRegressor
from tsfresh.utilities.dataframe_functions import impute

# Fix needed to pandas datareader
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import datetime

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm


start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2017, 1, 1)

# Need to use iex instead of google
x = web.DataReader("F", 'iex', start, end)
x.head()


x.drop("volume", axis=1).plot(figsize=(15, 6))



df_shift, y = make_forecasting_frame(x["high"], kind="price", max_timeshift=20, rolling_direction=1)

X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value", impute_function=impute,
                     show_warnings=False)
X = X.loc[:, X.apply(pd.Series.nunique) != 1]
X["feature_last_value"] = y.shift(1)


X = X.iloc[1:, ]
y = y.iloc[1: ]


ada = AdaBoostRegressor(n_estimators=10)
y_pred = [np.NaN] * len(y)

isp = 100  # index of where to start the predictions
assert isp > 0

for i in tqdm(range(isp, len(y))):
    print(i)


    ada.fit(X.iloc[:i], y[:i])
    y_pred[i] = ada.predict(X.iloc[i, :].values.reshape((1, -1)))[0]

y_pred = pd.Series(data=y_pred, index=y.index)













