from importlib import reload
import simfin
out = reload(simfin)
from simfin import *

# Extract and flaten simfin data set
if not os.path.isfile('tmp/extract.zip'):
    simfin = SimFin().extract().flatten()
else:
    simfin = SimFin().flatten()

# simfin = simfin.query(['FLWS','TSLA','A','AAPL','ADB','FB'])
# df = simfin.data_df

simfin = simfin.predict_rf_reg(lag=-1, max_depth=10, max_features="sqrt", min_samples_leaf=5, n_estimators=100)

# simfin = simfin.query(['A']).csv()

# Add target
work = 'tmp/target_reg'
if not os.path.isfile(work):
    simfin = simfin.target_reg(field='Revenues', lag=-1).save(work)
else:
    simfin = simfin.load(work)


# simfin.csv()
df = simfin.data_df

# Create model to predict target
work = 'tmp/xgboost'
if not os.path.isfile(work):
    simfin = sifmin.xgboost('').save(work)


# Add target
work = 'tmp/final_target'
if not os.path.isfile(work):
    simfin = simfin.target_class(field='Flat_SPQA', lag=-2).save(work)






