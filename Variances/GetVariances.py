from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
from Notebooks.LinkDatabases.PostComments import PostDataDatabase
import numpy as np

facebookDb = FacebookDataDatabase()
commentDb = PostDataDatabase()

commentCounts = facebookDb.selectColumnData("commentCount")
print("Comment Count Variance: {0}".format(np.var(commentCounts)))

shareCounts = facebookDb.selectColumnData("shareCount")
print("Share Count Variance: {0}".format(np.var(shareCounts)))

sentiments = list(map(lambda x: x[0] * 100, facebookDb.selectColumnData("postPositivity")))
print(sentiments[:20])
print("Sentiment Variance: {0}".format(np.var(sentiments)))
