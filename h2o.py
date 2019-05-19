import h2o
from h2o.automl import H2OAutoML
import pandas as pd

h2o.init()


self = self.target(field="Revenues", type='regression')
self.data_df.describe()
max(self.data_df['Target'])






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
x = hf_train.columns
y = "Target"
x.remove(y)

aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=hf_train, validation_frame=hf_val)

lb = aml.leaderboard
lb.head(rows=lb.nrows)

hf_test = h2o.H2OFrame(self.X_test)
hf_y = h2o.H2OFrame(self.y_test)
hf_y
pred = aml.predict(hf_test)
pred

