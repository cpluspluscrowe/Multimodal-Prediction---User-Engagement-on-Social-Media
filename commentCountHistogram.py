from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
import matplotlib.pyplot as plt

facebookDb = FacebookDataDatabase()
commentCounts = list(map(lambda x: x[0], facebookDb.selectColumnData("commentCount")))
plt.hist(commentCounts, bins=2000)  # arguments are passed to np.histogram
plt.xlim(0, 150)
plt.title("Histogram of Comment Counts")
plt.xlabel("Comment Count")
plt.ylabel("Count in Bin")
plt.savefig("/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/commentCountHistogram.png")
