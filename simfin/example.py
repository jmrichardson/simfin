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

# simfin = simfin.query(['FLWS','TSLA','A','AAPL','ADB','FB'])
# df = simfin.data_df

# Add predicted key features
# for feature in key_features:
    # log.info(f"Feature {feature} ...")
    # simfin = simfin.predict_rf(field=feature, lag=-1, type='reg', thresh=None, max_depth=10, max_features="sqrt", min_samples_leaf=5, n_estimators=100)
    # simfin = simfin.predict_rf(field=feature, lag=-1, type='class', thresh=None, max_depth=10, max_features="sqrt", min_samples_leaf=5, n_estimators=100)


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






