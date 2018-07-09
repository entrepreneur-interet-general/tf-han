import builtins
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from .experiment import Experiment
from .hyperparameters import HP

try:
    from IPython.lib import deepreload

    builtins.reload = deepreload.reload
except:
    pass
