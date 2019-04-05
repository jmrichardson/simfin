from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from loguru import logger as log
import re
import os
import sys
from importlib import reload

# Set current working directory (except for interactive shell)
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = 'd:/projects/quant/quant/simfin/mod_model'

# Extend path for local imports
os.chdir(cwd)
rootPath = re.sub(r"(.*quant).*", r"\1", cwd)
sys.path.extend([cwd, rootPath])

import simfin
# out = reload(simfin)
from simfin import *


df = SimFin().load('rf').data_df


import inspect
import simfin
[m[0] for m in inspect.getmembers(simfin, inspect.isclass) if m[1].__module__ == 'SimFin']