from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase

facebookDb = FacebookDataDatabase()

shareCountsTuples = facebookDb.selectColumnData("shareCount")
shareCounts = list(map(lambda x: x[0], shareCountsTuples))

import numpy as np
from matplotlib import pyplot as plt

# fixed bin size
bins = np.arange(0, 100, 1)  # fixed bin size

plt.xlim([min(shareCounts), 100])

plt.hist(shareCounts, bins=bins, alpha=0.5)

plt.savefig('/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/DataStats/shareCountHist.png')

print(np.std(shareCounts))
print(np.var(shareCounts))
