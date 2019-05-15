import pandas as pd
from scipy.stats import linregress
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
from sklearn import linear_model

facebookDb = FacebookDataDatabase()

shareCounts = list(map(lambda x: x[0], facebookDb.selectColumnData("shareCount")))
commentCounts = list(map(lambda x: x[0], facebookDb.selectColumnData("commentCount")))
data = list(zip(commentCounts, shareCounts))
df = pd.DataFrame(data, columns=["shareCount", "commentCount"])
df.to_csv("counts.csv")
