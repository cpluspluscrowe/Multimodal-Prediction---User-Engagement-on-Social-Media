from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
import pandas as pd
import numpy as np

fbDatabase = FacebookDataDatabase()
counts = list(map(lambda x: x[0] if x[0] > 0 else 0, fbDatabase.selectColumnData("commentCount")))
for x in counts[:100]:
    print(x)
df = pd.DataFrame(counts, columns=["commentCount"])
df.to_csv("comment_counts.csv")
