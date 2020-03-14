# need to run twice:
# https://github.com/ipython/ipython/issues/11098

import pandas as pd
from pandas import *

import scipy.stats as stats
from scipy.stats import *

from importlib import reload
from dask.distributed import Client, Future

import seaborn as sns
from seaborn import *

# %pylab inline
ipython = get_ipython()
ipython.run_line_magic('pylab', 'inline')
ipython.run_line_magic('config', "InlineBackend.figure_format='retina'")

# df
set_option('display.large_repr', 'info')
set_option('display.precision', 4)
DF, SR = DataFrame, Series

# plotting
sns.set('talk')
figsize(9, 6)
rcParams['axes.titlepad'] = 18
rcParams['legend.frameon'] = False
# rcParams['legend.borderaxespad'] = 0
# rcParams['legend.loc'] = (1.01,0)


# reset any bultin we may have overidden
import builtins
for var in dir(builtins):
    globals()[var] = getattr(builtins, var)
