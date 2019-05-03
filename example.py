from importlib import reload
import simfin
out = reload(simfin)
from simfin import *
from config import *
from loguru import logger as log

# Change log format
log.remove()
out = log.add(sys.stdout, format="<b>{time:YYYY-mm-dd hh:mm:ss}</b> - <green>{message}</green>")

# Extract and flaten simfin data set
if not os.path.isfile('tmp/extract.zip'):
    simfin = SimFin().extract().flatten()
else:
    simfin = SimFin().flatten()

# simfin = simfin.query(['FLWS','TSLA','A','AAPL','ADB','FB'])

# simfin = simfin.missing_rows()

# simfin = simfin.sample(frac=.1)

# simfin = simfin.features()

# simin = simfin.predict_features()

simfin = simfin.target()
simfin = simfin.split()
self = simfin
df = simfin.data_df










simfin = simfin.select_features()

simfin = simfin.tsf()
df = simfin.data_df

simfin = simfin.target(field='Flat_SPQA', type='class', lag=-1)

# simfin.catboost_target(init_learning_rate=.025, max_evals=50, eval_metric="Precision", od_wait=100, verbose=0)
# simfin.catboost_target(init_learning_rate=.05, max_evals=2, eval_metric="Precision", od_wait=10, verbose=1)
simfin = simfin.catboost_target(init_learning_rate=.025, max_evals=2, eval_metric="Precision", od_wait=10, verbose=0)





df = simfin.data_df
X = simfin.X

y = simfin.y
X_train = simfin.X_train
y_train = simfin.y_train
X_train_split = simfin.X_train_split
y_train_split = simfin.y_train_split
X_val_split = simfin.X_val_split
y_val_split = simfin.y_val_split
groups = simfin.groups
X_test = simfin.X_test
y_test = simfin.y_test


len(X_train) + len(y_test)



look = pd.Series(simfin.importances)

# df.to_pickle('tmp/df.pkl')




############################

# simfin = simfin.select_features(thresh=.05)











# Add target
work = 'tmp/final_target'
if not os.path.isfile(work):
    simfin = simfin.target(field='Flat_SPQA', type='class', lag=-1).save(work)
else:
    simfin = SimFin().load(work)


simfin = simfin.process()





#-------------- Example snips

# Add predicted key features
# for feature in key_features:
# log.info(f"Feature {feature} ...")
# simfin = simfin.random_forest(field=feature, lag=-1, type='reg', thresh=None, max_depth=10, max_features="sqrt", min_samples_leaf=5, n_estimators=100)
# simfin = simfin.random_forest(field=feature, lag=-1, type='class', thresh=None, max_depth=10, max_features="sqrt", min_samples_leaf=5, n_estimators=100)


# simfin.csv()
# df = simfin.data_df


search.save_model('tmp/model')
