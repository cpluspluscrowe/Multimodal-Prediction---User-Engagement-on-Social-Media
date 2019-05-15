import os
import sys

sys.path.append("../../")
import statistics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from Notebooks.LinkDatabases.PostComments import PostDataDatabase
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase

commentDb = PostDataDatabase()


class SentimentAnalyzer:
    analyser = SentimentIntensityAnalyzer()

    @staticmethod
    def GetPostSentiment(postId):
        def get_sentiment_scores(sentence):
            sentiment = SentimentAnalyzer.analyser.polarity_scores(sentence)
            compound = sentiment["compound"]
            if (sentiment["neu"] == 1) and sentiment["neg"] == 0 and sentiment["pos"] == 0 and (compound == 0):
                return 0
            else:
                return compound

        messages = commentDb.getMessages(postId)
        if len(messages) == 0:
            print("No comments for: {0}".format(postId))
            return
        scores = []
        for message in messages:
            sentiment = get_sentiment_scores(message[0])
            if sentiment != -1:
                scores.append(sentiment)
        if not scores:
            return 0
        return statistics.mean(scores)


facebookDb = FacebookDataDatabase()
post_ids = list(map(lambda x: x[0], facebookDb.getImageIdWithPositiveCommentCounts()))
comment_db_post_ids = list(map(lambda x: x[0], commentDb.getPostIds()))
post_ids_with_new_comments = set(post_ids).union(set(comment_db_post_ids))

for data in list(post_ids_with_new_comments):
    postId = data
    mean_sentiment_score = SentimentAnalyzer.GetPostSentiment(postId)
    print(mean_sentiment_score)
    if mean_sentiment_score:
        facebookDb.insertSentimentData(mean_sentiment_score, -1, postId)
