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
