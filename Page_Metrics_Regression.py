from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
import pandas as pd

facebookDb = FacebookDataDatabase()

# fanCount INT,numberOfRatings INT, talkingAboutCount INT, pageRating REAL
data = facebookDb.selectPageMetrics()
df = pd.DataFrame(data, columns=["fanCount", "numberOfRatings", "talkingAboutCount", "shareCount", "commentCount",
                                 "sentiment"])
df.to_csv("page_metrics.csv")
