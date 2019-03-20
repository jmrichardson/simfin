import h2o
from h2o.automl import H2OAutoML
h2o.init()

import pickle
from loguru import logger

# Set current directory
try:
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
except NameError:
    import os
    os.chdir('d:/projects/quant/model/h2o')


# Load SimFin dataset
logger.info("Loading dataset ...")
with open('../../quarterly/source/simfin/data_process.pickle', 'rb') as handle:
    data = pickle.load(handle)

hf = h2o.H2OFrame(data)

train, test = hf.split_frame(ratios=[.7])

x = train.columns
y = 'Share Price'
x.remove('Share Price')


aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=x, y=y, training_frame=train)

