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

# %pylab inline + other magic
ipython = get_ipython()
ipython.magic('pylab inline')
ipython.magic('config InlineBackend.figure_format="retina"')
ipython.magic('load_ext autoreload')
ipython.magic('autoreload 2')

# cells can have multiple outputs
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"'axes'axes

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

# copied from quanstats styling:
# https://github.com/ranaroussi/quantstats/blob/71fb349ddeb500ee65af3048c1bf9d481f20c415/quantstats/_plotting/core.py
sns.set(font_scale=1.5, rc={
    'figure.figsize': (9, 6),
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'grid.color': '#dddddd',
    'grid.linewidth': 0.5,
    "lines.linewidth": 1.5,
    'text.color': '#333333',
    'xtick.color': '#666666',
    'ytick.color': '#666666',
    'axes.titlepad': 18,
    'legend.frameon': False
})

# reset any bultin we may have overidden
import builtins
for var in dir(builtins):
    globals()[var] = getattr(builtins, var)
