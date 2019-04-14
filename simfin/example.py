from importlib import reload
import simfin
out = reload(simfin)
from simfin import *
from config import *


# Extract and flaten simfin data set
if not os.path.isfile('tmp/extract.zip'):
    simfin = SimFin().extract().flatten()
else:
    simfin = SimFin().flatten()

simfin = simfin.query(['FLWS','TSLA','A','AAPL','ADB','FB'])

simfin = simfin.missing_rows()
simfin = simfin.history()

simfin = simfin.tsf()


simfin = simfin.target(field='Flat_SPQA', type='class', lag=-1)

simfin = simfin.features()
df = simfin.data_df

# Add target
work = 'tmp/final_target'
if not os.path.isfile(work):
    simfin = simfin.target(field='Flat_SPQA', type='class', lag=-1).save(work)
else:
    simfin = SimFin().load(work)


simfin = simfin.process()

df = simfin.data_df




#-------------- Example snips

# Add predicted key features
# for feature in key_features:
# log.info(f"Feature {feature} ...")
# simfin = simfin.predict_rf(field=feature, lag=-1, type='reg', thresh=None, max_depth=10, max_features="sqrt", min_samples_leaf=5, n_estimators=100)
# simfin = simfin.predict_rf(field=feature, lag=-1, type='class', thresh=None, max_depth=10, max_features="sqrt", min_samples_leaf=5, n_estimators=100)


# simfin.csv()
# df = simfin.data_df


