from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
import numpy as np

facebookDb = FacebookDataDatabase()

commentCountsTuples = facebookDb.selectColumnData("commentCount")
commentCounts = list(map(lambda x: x[0], commentCountsTuples))
commentCountsLog = list(map(lambda x: np.log(x) if x > 0 else x, commentCounts))

import numpy as np
from matplotlib import pyplot as plt

# fixed bin size
bins = np.arange(0, 100, 1)  # fixed bin size

plt.xlim([min(commentCountsLog), 100])

plt.hist(commentCountsLog, bins=bins, alpha=0.5)

plt.savefig('/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/DataStats/commentCountHist.png')

print(np.std(commentCountsLog))
print(np.var(commentCountsLog))
