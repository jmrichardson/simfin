import pandas as pd
import numpy as np
from loguru import logger as log
from config import *


def get_weight_ffd(d, thres, lim):
    w, k = [1.], 1
    ctr = 0
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
        ctr += 1
        if ctr == lim - 1:
            break
    w = np.array(w[::-1]).reshape(-1, 1)
    return w


def frac_diff_ffd(x, d, thres=1e-5):
    w = get_weight_ffd(.5, thres, min_quarters)
    w = get_weight_ffd(d, thres, min_quarters)
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    for i in range(width, len(x)):
        s = x[i - width:i + 1]
        print(s)
        print(w.T)
        dp = np.dot(w.T, s)[0]
        print(dp)
        output.append(dp)
    return np.array(output)
x = pd.Series(frac_diff_ffd(df[field].apply(np.log), d=0.4, thres=1e-4))


3.955585 * -.4



# from statsmodels.tsa.stattools import adfuller
# adfuller(x, 12)


def by_ticker(df, field, lag, thresh):

    ticker = str(df['Ticker'].iloc[0])

    # Fractional diff
    x = pd.Series(frac_diff_ffd(df[field].apply(np.log), d=0.4, thres=1e-4))

    # Pct Diff
    df.loc[:, 'Target'] = (x.shift(lag) - x) / x

    # Replace inf values with nan (Target field)
    df['Target'] = df['Target'].replace([np.inf, -np.inf], np.nan)

    # y = df[field].apply(np.log)
    y = df[field]
    pct_diff = (y.shift(lag) - y) / y
    df['pct_diff'] = pct_diff
    df[['Flat_SPQA', 'pct_diff', 'Target']]

    # If thresh, then 1 if > thresh, else 0
    if thresh is not None:
        df[field_name] = df[field_name].map(lambda x: 1 if x >= thresh else (np.nan if np.isnan(x) else 0))
    # else:
        # df[field_name] = df[field_name].map(lambda x: 1 if x > 0 else (np.nan if np.isnan(x) else 0))

    # Class 0 or 1:  < 0 -> 0, > 1 -> 1
    # df[field_name] = df['Target'].where(df['Target'] > 0, other=0)
    # df[field_name] = df['Target'].where(df['Target'] <= 0, other=1)

    return df


class Target:

    def target(self, field='Flat_SPQA', lag=-1, thresh=None):

        log.info(f"Adding target {field} ...")

        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker, field, lag, thresh)

        self.data_df.reset_index(drop=True, inplace=True)

        # Remove null target rows and sort by date
        # self.data_df = self.data_df[pd.notnull(self.data_df['Target'])]

        # print(max(self.data_df["Target"]))

        return self


