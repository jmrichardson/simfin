from importlib import reload
import simfin
out = reload(simfin)
from simfin import *

# Extract and flaten simfin data set
if not os.path.isfile('tmp/extract.zip'):
    simfin = SimFin().extract().flatten()
else:
    simfin = SimFin().flatten()

df = simfin.data_df

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






