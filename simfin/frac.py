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
    w = get_weight_ffd(d, thres, min_quarters)
    width = len(w) - 1
    output = []
    output.extend([0] * width)
    for i in range(width, len(x)):
        output.append(np.dot(w.T, x[i - width:i + 1])[0])
    return np.array(output)



# from statsmodels.tsa.stattools import adfuller

# adfuller(x, 12)
# adfuller(fracs, 12)
# adf[0]

# import matplotlib as plt
# import matplotlib.pyplot as plt

# plt.plot(x)
# plt.show()

# plt.interactive(False)
# x.plot()

def by_ticker(df, field="Share Price", lag=-1, thresh=None):

    # field = "Share Price"
    # type = "reg"
    # lag = -1
    # thresh = None

    # x = pd.Series(np.random.random(10000))
    # x = df[field].apply(np.log)
    # x = df[field]
    # d = 0.4
    # thres=1e-4


    ticker = str(df['Ticker'].iloc[0])
    # log.info(f"Adding target field {field} for {ticker}...")

    # fracs = frac_diff_ffd(x, d=0.4, thres=1e-4)
    # fracs = frac_diff_ffd(df[field].apply(np.log), d=0.4, thres=1e-5)
#
    # fracDiff_FFD(x, d=0.4, thres=1e-5)
#
    # a = pd.DataFrame(data=np.transpose([np.array(fracs), close['Close'].values]),
                     # columns=['Fractional differentiation FFD', 'SP500'])


    # Fractional (memory of min_quarters)
    x = pd.Series(frac_diff_ffd(df[field].apply(np.log), d=0.4, thres=1e-4))

    # Lag
    x_lag = pd.Series(x).shift(lag)

    # Pct Diff
    df.loc[:, 'Target'] = (x_lag - x) / x

    # Remove any inf values (Target field)
    df['Target'] = df['Target'].replace([np.inf, -np.inf], np.nan)

    # Sanity check (temporary)
    if max(df["Target"]) > 100000:
        print(max(df["Target"]))
        print(f"{ticker}")
        assert("wtf")

    # Remove rows where target is nan
    # df = df[pd.notnull(df[field_name])]

    # If thresh, then 1 if > thresh, else 0
    # if type == 'class':
        # if thresh is None:
            # df[field_name] = df[field_name].map(lambda x: 1 if x > 0 else (np.nan if np.isnan(x) else 0))
        # else:
            # df[field_name] = df[field_name].map(lambda x: 1 if x >= thresh else (np.nan if np.isnan(x) else 0))


    # Class 0 or 1:  < 0 -> 0, > 1 -> 1
    # df[field_name] = df['Target'].where(df['Target'] > 0, other=0)
    # df[field_name] = df['Target'].where(df['Target'] <= 0, other=1)

    return df


class Target:

    def target(self, field='Flat_SPQA', type='class', lag=-1, thresh=None):

        log.info(f"Adding target {field} ...")

        self.data_df = self.data_df.groupby('Ticker').apply(by_ticker, field, type, lag, thresh)

        self.data_df.reset_index(drop=True, inplace=True)

        # Remove null target rows and sort by date
        # self.data_df = self.data_df[pd.notnull(self.data_df['Target'])]

        # print(max(self.data_df["Target"]))

        return self


