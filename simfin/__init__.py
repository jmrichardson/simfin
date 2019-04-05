import pandas as pd
from loguru import logger as log
import re
import os
import sys
import pickle
from importlib import reload

# Set current working directory (except for interactive shell)
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = 'd:/projects/quant/quant/simfin'

# Extend path for local imports
os.chdir(cwd)
rootPath = re.sub(r"(.*quant).*", r"\1", cwd)
sys.path.extend([rootPath, cwd])

# Import helper modules - FORCE RELOAD DURING TESTING - REMOVE THIS
# import config
# out = reload(config)
# from config import *

