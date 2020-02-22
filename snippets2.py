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
    """
    eg:
    
        def func(x, y, z):
        return (x*2, y, z)

        grid_kwargs = {
            'x': [1,2,3], 
            'y': ['a', 'b'], 
            'z': [
               {'arg1':1, 'arg2': 2},
               {'arg1':2, 'arg2': 3}
            ]
        }

        grid = run_grid(func, c, **grid_kwargs)
        
    grid:
    
        x  y  z                     
        1  a  {'arg1': 1, 'arg2': 2}    <Future: status: finished, type: tuple, key: f...
              {'arg1': 2, 'arg2': 3}    <Future: status: finished, type: tuple, key: f...
           b  {'arg1': 1, 'arg2': 2}    <Future: status: finished, type: tuple, key: f...
              {'arg1': 2, 'arg2': 3}    <Future: status: finished, type: tuple, key: f...
        2  a  {'arg1': 1, 'arg2': 2}    <Future: status: finished, type: tuple, key: f...
              {'arg1': 2, 'arg2': 3}    <Future: status: finished, type: tuple, key: f...
           b  {'arg1': 1, 'arg2': 2}    <Future: status: finished, type: tuple, key: f...
              {'arg1': 2, 'arg2': 3}    <Future: status: finished, type: tuple, key: f...
        3  a  {'arg1': 1, 'arg2': 2}    <Future: status: finished, type: tuple, key: f...
              {'arg1': 2, 'arg2': 3}    <Future: status: finished, type: tuple, key: f...
           b  {'arg1': 1, 'arg2': 2}    <Future: status: finished, type: tuple, key: f...
              {'arg1': 2, 'arg2': 3}    <Future: status: finished, type: tuple, key: f...
    
    grid.loc[3, 'a', :].map(Future.result):
        x  y  z                     
        3  a  {'arg1': 1, 'arg2': 2}    (6, a, {'arg1': 1, 'arg2': 2})
              {'arg1': 2, 'arg2': 3}    (6, a, {'arg1': 2, 'arg2': 3})
    """
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
