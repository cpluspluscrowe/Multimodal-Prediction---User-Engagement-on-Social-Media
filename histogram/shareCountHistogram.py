from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
import matplotlib.pyplot as plt

facebookDb = FacebookDataDatabase()
commentCounts = list(map(lambda x: x[0], facebookDb.selectColumnData("shareCount")))[:5000]
plt.hist(commentCounts, bins=5000)  # arguments are passed to np.histogram
plt.xlim(0, 200)
plt.title("Histogram of Share Counts")
plt.xlabel("Share Count")
plt.ylabel("Count")
plt.savefig("/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/shareCountHistogram.png")
