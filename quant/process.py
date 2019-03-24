import talib
import numpy as np
from loguru import logger as log

def yo():
    print('yo')

# Add momentum indicator on features
def mom(df, features, count):
    for feature in features:
        for i in range(1, count+1, 1):
            try:
                df[feature + ' Mom ' + str(i) + 'Q'] = talib.MOM(np.array(df[feature]), i)
            except:
                log.warning("Unable to add momentum for: " + feature)
                pass
    return df


# Calculate trailing twelve months
def TTM(df, features, roll, days):
    def lastYearSum(series):
        # Must have x previous inputs
        if len(series) < roll:
            return np.nan
        # Must be within X days date range
        else:
            firstDate = df['Date'][series.head(1).index.item()]
            lastDate = df['Date'][series.tail(1).index.item()]
            if (lastDate - firstDate).days > days:
                return np.nan
            else:
                return series.sum()
    for feature in features:
        try:
            df[feature + ' TTM'] = df[feature].rolling(roll, min_periods=1).apply(lastYearSum, raw=False)
        except:
            log.warning("Unable to add TTM for: " + feature)
        pass
    return df


