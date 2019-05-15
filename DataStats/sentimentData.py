from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase

facebookDb = FacebookDataDatabase()

sentimentsTuples = facebookDb.selectColumnData("postPositivity")
sentiments = list(map(lambda x: x[0], sentimentsTuples))

import numpy as np
from matplotlib import pyplot as plt

# fixed bin size
bins = np.arange(0, 100, 1)  # fixed bin size

plt.xlim([min(sentiments), 100])

plt.hist(sentiments, bins=bins, alpha=0.5)

plt.savefig('/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/DataStats/sentimentHist.png')

print(np.std(sentiments))
print(np.var(sentiments))
