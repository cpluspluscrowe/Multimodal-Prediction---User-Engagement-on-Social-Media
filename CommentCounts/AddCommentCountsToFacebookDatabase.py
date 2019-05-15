import pandas as pd

from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
from Notebooks.LinkDatabases.PostComments import PostDataDatabase

facebookDb = FacebookDataDatabase()
commentDb = PostDataDatabase()

comment_data = commentDb.selectPostData()

df = pd.DataFrame.from_records(comment_data, columns=["imageId", "commentId", "text"])

for imageId in df.imageId.unique():
    count = len(df[df["imageId"] == imageId])
    facebookDb.insertCommentCountData(count, imageId)
