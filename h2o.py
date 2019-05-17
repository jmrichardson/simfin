import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init()

feature = "Revenues"

self = self.target(field=feature, type='regression')
self = self.process()
self = self.split()

# Create h2o training frame
df = pd.concat([self.X_train, pd.DataFrame(self.y_train)], axis=1)
df.columns.values[-1] = "Target"
hf_train = h2o.H2OFrame(df)

# Create h2o validation frame
df = pd.concat([self.X_val, pd.DataFrame(self.y_val)], axis=1)
df.columns.values[-1] = "Target"
hf_val = h2o.H2OFrame(df)

# Identify predictors and response
x = hf.columns
y = "Target"
x.remove(y)

aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=hf_train, validation_frame=hf_val)

lb = aml.leaderboard
lb.head(rows=lb.nrows)



