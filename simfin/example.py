from importlib import reload
import simfin
out = reload(simfin)
from simfin import *

# Extract and flaten simfin data set
if not os.path.isfile('tmp/extract.zip'):
    simfin = SimFin().extract().flatten()
else:
    simfin = SimFin().flatten()



# Create model to predict target
work = 'tmp/xgboost'
if not os.path.isfile(work):
    simfin = sifmin.xgboost('').save(work)


# Add target
work = 'tmp/example'
if not os.path.isfile(work):
    simfin = simfin.target_class().save(work)




