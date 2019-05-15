import os
import sqlite3

file_location = os.path.dirname(os.path.realpath(__file__))
db_location = sqlite3.connect('{0}/AdData.db'.format(file_location))


class PostDataDatabase():
    def __init__(self):
        conn = db_location
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS PostData (imageId TEXT, commentId TEXT, comment TEXT);''')
        conn.commit()
        c.close()
        #

    def getDistinctPostCount(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT imageId FROM PostData")
        results = c.fetchall()
        c.close()
        if results:
            return len(results)

    def insertPostData(self, imageId, commentId, commentText):
        conn = db_location
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO PostData(imageId, commentId, comment) 
            values('{0}','{1}','{2}');'''.format(
            imageId, commentId, commentText))
        conn.commit()
        c.close()

    def getPostIds(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT imageId FROM PostData")
        results = c.fetchall()
        c.close()
        return results

    def selectPostData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT * FROM PostData")
        results = c.fetchall()
        c.close()
        return results

    def getMessages(self, postId):
        conn = db_location
        c = conn.cursor()
        c.execute("select distinct comment from PostData where imageId = '{0}'".format(postId))
        results = c.fetchall()
        c.close()
        return results

    def deletePostData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("DELETE FROM PostData WHERE 1 = 1")
        conn.commit()
        c.close()

    def dropPostDataTable(self):
        conn = db_location
        c = conn.cursor()
        c.execute("DROP TABLE PostData")
        conn.commit()
        c.close()

    def getCommentCount(self, imageId):
        conn = db_location
        c = conn.cursor()
        c.execute("select distinct commentId from PostData where imageId = '{0}'".format(imageId))
        results = c.fetchall()
        c.close()
        if results:
            return len(results)

    def isCommentInDb(self, post_key):
        conn = db_location
        c = conn.cursor()
        c.execute("select imageId from FacebookData where commentId like '{0}_%' limit 1".format(page_key))
        results = c.fetchall()
        c.close()
        return len(results) > 0
