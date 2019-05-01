from autofeat import AutoFeatRegression


model = AutoFeatRegression()
df = model.fit_transform(X, y)



# predict the target for new test data points
y_pred = model.predict(X_test)
# compute the additional features for new test data points
# (e.g. as input for a different model)
df_test = model.transform(X_test)



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autofeat import AutoFeatRegression



# generate some toy data
np.random.seed(15)
x1 = np.random.rand(1000)
x2 = np.random.randn(1000)
x3 = np.random.rand(1000)
target = 2 + 15*x1 + 3/(x2 - 1/x3) + 5*(x2 + np.log(x1))**3
X = np.vstack([x1, x2, x3]).T
# autofeat!
afreg = AutoFeatRegression()
df = afreg.fit_transform(pd.DataFrame(X, columns=["x 1", "x.2", "x/3"]), target)
#df = afreg.fit_transform(X, target)