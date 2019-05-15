from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
import matplotlib.pyplot as plt

facebookDb = FacebookDataDatabase()
commentCounts = list(filter(lambda x: x > -1, map(lambda x: x[0], facebookDb.selectColumnData("postPositivity"))))
plt.hist(commentCounts, bins=100)  # arguments are passed to np.histogram
plt.title("Histogram of Comment Sentiment Positivity")
plt.xlabel("Count Sentiment")
plt.ylabel("Count")
plt.savefig("/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/commentSentimentHistogram.png")
