import pandas as pd


def run_grid(func, grid_kwargs, client=None):
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
        
        client = Client()
        
        grid = run_grid(func, grid_kwargs, client)
        
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
    def _make_hashable(o):
        try:
            hash(o)
            return o
        except TypeError:
            return str(o)

    param_grid = ParameterGrid(grid_kwargs)
    hashable_param_grid = [{k: _make_hashable(v) for k, v in d.items()} for d in param_grid]

    if client is not None:
        results = [client.submit(func, **params) for params in param_grid]
    
    else:
        results = [func(**params) for params in param_grid]
    
    idx = pd.MultiIndex.from_frame(pd.DataFrame(hashable_param_grid))
    grid = pd.Series(data=results, index=idx)

    return grid
