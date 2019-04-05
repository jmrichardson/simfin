import re
import os
import sys
from importlib import reload

# Set current working directory (except for interactive shell)
try:
    cwd = os.path.dirname(os.path.realpath(__file__))
except:
    cwd = 'd:/projects/simfin/simfin'

# Extend path for local imports
os.chdir(cwd)
rootPath = re.sub(r"(.*simfin).*", r"\1", cwd)
sys.path.extend([rootPath, cwd, cwd + '/mod_data', cwd + '/mod_model'])

import sf
out = reload(sf)
from sf import *

