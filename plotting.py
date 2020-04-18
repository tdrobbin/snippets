sns.set('talk', style='ticks')
sns.set_palette('Paired')
ax = DataFrame(randn(7, 3),).plot(kind='bar').legend(bbox_to_anchor=(1.2, .7), borderaxespad=0, frameon=False)
plt.xticks(rotation=45, horizontalalignment='right')
plt.title('Some Data 1', pad=30)
sns.despine()

sns.set('talk', style='ticks')
sns.set_palette('Paired')
ax = DataFrame(randn(1000, 3)).cumsum().plot().legend(bbox_to_anchor=(1.2, .7), borderaxespad=0, frameon=False)
plt.xticks(rotation=45, horizontalalignment='right')
plt.title('Some Data 1', pad=30)
sns.despine()

sns.set('talk')
ax = DataFrame(randn(7, 3),).plot(kind='bar', title='Some Data 2')
plt.xticks(rotation=45, horizontalalignment='right')
plt.title('Notebook', pad=18)

sns.set('talk', style='whitegrid')
ax = DataFrame(randn(7, 3),).plot(kind='bar', title='Some Data 2')
plt.xticks(rotation=45, horizontalalignment='right')
plt.title('Notebook', pad=18)

# regular seaborn style legend on upper right, middle right, bottom
ax = DF(randn(30,3)).cumsum().plot(title='Random Data').legend(bbox_to_anchor=(1.02,1), loc="upper left", borderaxespad=0, frameon=False)
ax = DF(randn(30,3)).cumsum().plot(title='Random Data').legend(bbox_to_anchor=(1.02,.5), loc="center left", borderaxespad=0, frameon=False)

def theme_plots(dark=False):
  plt.style.use('default')
  from qbstyles import mpl_style
  mpl_style(dark=dark)
  
  if dark:
    plt.rcParams['figure.facecolor'] = '#111111'

def set_plotting_style(dark=False):
    if dark:
        sns.set(
            context='talk' ,
            rc={
                'figure.figsize': (9, 6),
                'axes.facecolor': '#111111',
                'figure.facecolor': '#111111',
                'grid.color': '#6f6f6f',
                'grid.linewidth': 0.5,
                "lines.linewidth": 1.5,
                'text.color': 'white',
                'xtick.color': 'white',
                'ytick.color': 'white',
                'axes.titlepad': 18,
                'legend.frameon': False,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.bottom': False,
                'axes.spines.left': False,
                'patch.force_edgecolor': False,
                'axes.labelcolor': 'white'
            }
        )
    
    else:
        sns.set(
            context='talk' ,
            rc={
                'figure.figsize': (9, 6),
                'axes.facecolor': 'white',
                'figure.facecolor': 'white',
                'grid.color': '#dddddd',
                'grid.linewidth': 0.5,
                "lines.linewidth": 1.5,
                'text.color': 'black',
                'xtick.color': 'black',
                'ytick.color': 'black',
                'axes.titlepad': 18,
                'legend.frameon': False,
                'axes.spines.top': False,
                'axes.spines.right': False,
                'axes.spines.bottom': False,
                'axes.spines.left': False,
                'patch.force_edgecolor': False,
                'axes.labelcolor': 'black'
            }
        )
