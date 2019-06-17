# initialize notebooks
ipython = get_ipython()
ipython.run_line_magic('pylab', 'inline')
ipython.run_line_magic('config', "InlineBackend.figure_format='retina'")
import pandas as pd
from pandas import *
import seaborn as sns
sns.set()
from importlib import reload






# plotting

def _plotting_style(dark=False, scale_figsize=1.5):
    if dark:
        plt.style.use('dark_background')
    else:
        sns.set()
    
    plt.rcParams['figure.figsize'] = [6*(np.sqrt(scale_figsize)), 4*(np.sqrt(scale_figsize))]

    
def legend_outside(ax, position='right', **kwargs):
    if position == 'right':
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])
        ax.legend(loc='upper center', bbox_to_anchor=(1.1, 1), shadow=False, ncol=1, facecolor='white', edgecolor='white')
    
    elif position == 'bottom':
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=4, facecolor='white', edgecolor='white')

def pretty_plot(pdobj, percent=False, decimals=0, commas=False, save=False, **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    
    plt.style.use('seaborn-whitegrid')
    
    import seaborn as sns
    sns.set_palette('Paired')
    
    plt.rcParams.update({
        'axes.spines.right': False,
        'axes.spines.top': False,
        'axes.spines.left': False,
        'axes.spines.left': True,
        'font.size': 15,
        'figure.figsize': (15, 7),
        'axes.grid': False,
        'figure.dpi': 300.0
    })
    
    plt.figure()
    
    ax = pdobj.plot(**kwargs)
    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)
    
    if percent:
        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=decimals, xmax=1))
    
    else:
        if commas:
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    if save != False:
        plt.savefig(save)
        print('figure saved: {save}'.format(save=save))
    
    return ax

def annot_heatmap(df):
    corr = df
    with sns.set_context(style="white"):
        size = (len(corr) / 1.7, len(corr)*(9/11) / 1.7)
        f, ax = plt.subplots(figsize=size)

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)

        # with sns.set_context(context='white'):
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(corr.round(1), cmap=cmap, annot=True,
                    square=True, linewidths=0, cbar_kws={"shrink": .5})

def pretty_heatmap(df):
    corr = df
    corr = corr.applymap(lambda x: round(x, 2))

    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # ax = sns.heatmap(corr, square=True, cmap="Blues", annot=True, mask=mask, annot_kws={"size": 10})

    ax = sns.heatmap(corr, square=True, cmap="Blues", annot=True, mask=mask)

    # ax = sns.heatmap(corr, square=True, cmap="Blues", annot=True)


    # sns.despine(ax=ax)

    ax.tick_params(axis='x', pad=10)
    ax.tick_params(axis='y', pad=10)

    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
 
def plot_two_sr(w):
     with plt.rc_context(rc={'figure.figsize': (16, 5)}):
        f, ax = subplots(1,2)
        Series(w).plot(kind='bar', ax=ax[0])
        Series(w).cumsum().plot(kind='bar', ax=ax[1])


# dask
def make_return_tuple(func):
    @wraps(func)
    def new_func(**kwargs):     
        return (tuple(sorted(((k, v) for k, v in kwargs.items()), key=lambda x: x[0])), func(**kwargs))
    
    return new_func


def dask_get_ddclient_dashboard_address(ddclient):
    html_repr = ddclient._repr_html_()
    # import pdb; pdb.se
    match = re.search('>http.*<', html_repr)
    if match:
        return match.group().replace('>', '').replace('<', '')
    
    return 
    # return print(ddclient)


from distributed import Client

def dask_compute_grid(ddclient=None, func=None, **kwargs):
    temp_cluster = False
    completed = []
    
    if ddclient is None:
        print('creating local dask distributed cluster...')
        # ddclient = Client()
        ddclient = Client()

        temp_cluster = True
#         print('cluster dashboard available at: ' + get_ddclient_dashboard_address(ddclient))
    
    try:
        print('cluster dashboard available at: ' + dask_get_ddclient_dashboard_address(ddclient))
        from IPython.display import display
        display(ddclient)
        tfunc = make_return_tuple(func)
        kwargs_list = ([(k, i) for i in v] for k, v in kwargs.items())
        
        # tuple of cartesian products of {{(arg_name, arg_val) | arg_val in arg_vals} | arg_name in arg_names}
        cart_prod_tup = product(*kwargs_list)
        cart_prod_dicts = [dict(i) for i in cart_prod_tup]

        print('submitting {} jobs to cluster...'.format(len(cart_prod_dicts)))
        futures = [ddclient.submit(tfunc, **kwargs) for kwargs in cart_prod_dicts]

        print('computing jobs...')
        completed = ddclient.gather(futures)

        print('computation done')
    
    finally:
        if temp_cluster:
            print('shutting down cluster...')
            ddclient.close()
    
    print('done')
    return completed


def dask_submit_jobs_to_cluster(ddclient, func=None, **kwargs):
        from IPython.display import display
        display(ddclient)
        tfunc = make_return_tuple(func)
        kwargs_list = ([(k, i) for i in v] for k, v in kwargs.items())
        
        # tuple of cartesian products of {{(arg_name, arg_val) | arg_val in arg_vals} | arg_name in arg_names}
        cart_prod_tup = product(*kwargs_list)
        cart_prod_dicts = [dict(i) for i in cart_prod_tup]

        print('submitting {} jobs to cluster...'.format(len(cart_prod_dicts)))
        futures = [ddclient.submit(tfunc, **kwargs) for kwargs in cart_prod_dicts]

        return futures
 
 
 # other utils
 import pandas as pd
import numpy as np
from functools import reduce
import statsmodels.api as sm
from importlib import import_module
from pandas import DataFrame, Series

from collections import OrderedDict

from functools import wraps
from itertools import product
from time import sleep
# from dask.distributed import Client
from contextlib import contextmanager

import re
import pickle
import uuid
import git
import datetime
import os
import json 

# from .settings import data_base_dir

import uuid
import datetime

import pandas.core.datetools as dt

def get_uid(bits=64):
    # return str(uuid.uuid4().hex)
    return str(uuid.uuid4().hex)[:int(bits / 8)]


def get_now(as_str=False):
    now = datetime.datetime.now()
    if as_str:
        return dt.format(now)

    return now
    # return datetime.datetime.now()


def get_today(as_str=False):
    now = get_now()
    today = datetime.datetime(now.year, now.month, now.day)
    if as_str:
        return dt.format(today)
    
    return today


def format_date(date):
    f = dt.format(date)
    return '{}-{}-{}'.format(f[:4], f[4:6], f[6:])


def remove_time_from_datetime(date):
    return datetime.datetime(date.year, date.month, date.day)


def full_path(path):
    from os.path import realpath, abspath, expanduser, expandvars
    abspath(realpath(expandvars(expanduser('~/sdarsch/datacrix2'))))


def reindex_all(*dfs):
    """
    reindex all dataframes to intersection of all dataframes
    Parameters
    ----------
    dfs: list of dataframes as separate args

    Returns
    -------
    dfs in same order they were passed in but all reindexed
    """
    idx_intersection = reduce(np.intersect1d, (df.index for df in dfs))
    return (df.reindex(idx_intersection) for df in dfs)


def common_index_and_columns(dfs):
    idx_intersection = reduce(np.intersect1d, (df.index for df in dfs))
    col_intersection = reduce(np.intersect1d, (df.columns for df in dfs))

    return idx_intersection, col_intersection


def univar_regr(y, x):
    """
    removes nans and reindexes to common index
    Parameters
    ----------
    y: y var series
    x: x var series

    Returns
    -------
    [alpha, beta]
    """
    y, x = y.dropna(), x.dropna()
    y, x = reindex_all(y, x)
    Y, X = y.as_matrix(), x.as_matrix()

    X = sm.add_constant(X)
    
    try: 
        model = sm.OLS(Y, X)

        mdl_fit = model.fit()
        # import pdb; pdb.set_trace()

        params = pd.Series(mdl_fit.params, index=['alpha', 'beta'])
        params.name = 'coefficient'
        tvalues = pd.Series(mdl_fit.tvalues, index=['alpha', 'beta'])

        result = pd.DataFrame(params)
        result['tstat'] = tvalues
    
    except RuntimeError:
        result = pd.DataFrame(index=['alpha', 'beta'], columns=['coefficient','tstat'])

    return result


def clean_timeseries(srs, trim=.01, interpolate=True):
    '''
    will remove (1 - trim) percent of the data
    '''

    # import pdb; pdb.set_trace()
    srs = srs.copy()
    # orig_index = srs.index

    srs = srs.loc[srs.first_valid_index():]

    if trim is not None:
        # perc_to_trim = 1 - trim
        perc_to_trim = trim

        high = 1 - (float(perc_to_trim) / 2)
        low = 1 - high
        # import pdb; pdb.set_trace()

        quant_srs = srs.quantile([low, high])
        srs[(srs > quant_srs[high]) | (srs < quant_srs[low])] = np.nan

    if interpolate:
        srs = srs.interpolate()
    
    # srs = srs.reindex(orig_index)

    # srs = srs.dropna()

    return srs


def clean_df(df, trim=.01, interpolate=True, trim_on_pct_change=True):
    if trim_on_pct_change:
        df_rets = df.pct_change()
    
    else:
        df_rets = df.copy()
    # trimmed = df.apply(lam)
    df_rets_trimmed = df_rets.apply(lambda x: clean_timeseries(x, trim=trim, interpolate=False), axis=0)
    df_rets_trimmed_mask = df_rets_trimmed.notnull()

    df_orig_trimmed = df[df_rets_trimmed_mask]

    df_orig_trimmed_and_interpolated = df_orig_trimmed.apply(lambda x: x.interpolate())

    return df_orig_trimmed_and_interpolated


def shift_forward_one_day(view):
    view = view.copy()

    last_day = view.index[-1]
    new_last_day = last_day + dt.Day()
    view = view.reindex(view.index.append(pd.Index([new_last_day])))
    view = view.shift(1)

    return view


def get_srs_w_means(srs, means=[30, 60, 90]):
    # import pdb; pdb.set_trace()
    df = DataFrame({'rolling_' + str(m): srs.rolling(m, m).mean() for m in means})
    df.insert(loc=0, column='value', value=srs.copy())
    return df


def get_srs_w_vols(srs, vols=[30, 60, 90], annualize=True):
    # import pdb; pdb.set_trace()
    df = DataFrame({'rolling_' + str(m): srs.rolling(m, m).std() * (np.sqrt(365 / m) if annualize else 1) for m in vols})
    df.insert(loc=0, column='value', value=srs.copy())
    return df


def plot_w_means(srs, means=[30, 60, 90], ax=None, title='', figsize=(15, 5)):
    df = DataFrame({'rolling_' + str(m): srs.rolling(m, m).mean() for m in means})
    df.insert(loc=0, column='value', value=srs.copy())
    
    # if ax is not None:
    df.plot(ax=ax, title=title, figsize=figsize)
    
    
def reindex_to_mask(df, mask, fillna_zero=True):
    '''
    takes a dataframe and reindexes to a masks index and columns
    then fills any nans from the mask with zeros and sets anything
    outside the mask to nan
    '''

    df2 = df.copy()
    df2 = df2.reindex(index=mask.index, columns=mask.columns)
    
    if fillna_zero:
        df2 = df2.fillna(0.)[mask]
    else:
        df2 = df2[mask]
    
    return df2
    
def search(l, s):
    matched = []
    for i in l:
        if i.find(s) != -1:
            matched.append(i)
    
    return matched
