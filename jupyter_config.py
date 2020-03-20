# need to run twice:
# https://github.com/ipython/ipython/issues/11098

# everything should word in the standard base conda environment

import pandas as pd
from pandas import *

import scipy.stats as stats
from scipy.stats import *

from importlib import reload
from dask.distributed import Client, Future

import seaborn as sns
from seaborn import *

import matplotlib.pyplot as plt
import os

_init_cwd = os.getcwd()

# %pylab inline
ipython = get_ipython()
ipython.run_line_magic('pylab', 'inline')
ipython.run_line_magic('config', "InlineBackend.figure_format='retina'")

# cells can have multiple outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# df
# set_option('display.large_repr', 'info')
set_option('display.precision', 4)
DF, SR = DataFrame, Series

# plotting
sns.set('talk')
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['axes.titlepad'] = 18
plt.rcParams['legend.frameon'] = False
# plt.rcParams['legend.borderaxespad'] = 0
# plt.rcParams['legend.loc'] = (1.01, 0)

# reset any bultin we may have overidden
import builtins
for var in dir(builtins):
    globals()[var] = getattr(builtins, var)
