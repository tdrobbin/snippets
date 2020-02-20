# imports and configs
import pandas as pd
from pandas import *

import scipy.stats as stats
from scipy.stats import *

pd.set_option('display.large_repr', 'info')
pd.set_option('display.precision', 4)

# %pylab inline
ipython = get_ipython()
ipython.run_line_magic('pylab', 'inline')
ipython.run_line_magic('config', "InlineBackend.figure_format='retina'")

from numpy import array, test, testing

import seaborn as sns
sns.set()

from importlib import reload

from dask.distributed import Client, Future


def run_grid(func, client, **grid_kwargs):
    import pandas as pd
    
    def _make_hashable(o):
        try:
            hash(o)
            return o
        except TypeError:
            return str(o)

    keys = grid_kwargs.keys()
    vals = grid_kwargs.values()
    hasahble_vals = [[_make_hashable(o) for o in val] for val in vals]

    idx = pd.MultiIndex.from_product(hasahble_vals, names=keys)
    futures = [client.submit(func, **dict(zip(keys, args))) for args in idx]

    grid = pd.Series(data=futures, index=idx)
    
    return grid
