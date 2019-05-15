import os
import sqlite3
import unicodedata

file_location = os.path.dirname(os.path.realpath(__file__))
db_location = sqlite3.connect('{0}/AdData.db'.format(file_location))


class FacebookDataDatabase():
    def __init__(self):
        conn = db_location
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS FacebookData (imageId TEXT PRIMARY KEY, imageUrl TEXT, message TEXT, shareCount INT, commentCount INT,
          fanCount INT,numberOfRatings INT, talkingAboutCount INT, pageRating REAL, postPositivity REAL, postSubjectivity REAL)
        ;''')
        conn.commit()
        c.close()

    def get_post_ids(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT imageId FROM FacebookData")
        results = c.fetchall()
        c.close()
        if results:
            return results

    def get_message(self, imageId):
        conn = db_location
        c = conn.cursor()
        c.execute("select distinct message from FacebookData where imageId = '{0}'".format(imageId))
        results = c.fetchall()
        c.close()
        if results:
            return results[0][0]

    def getTextData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT imageId, message FROM FacebookData")
        results = c.fetchall()
        c.close()
        if results:
            return results

    def getSentiment(self, imageId):
        conn = db_location
        c = conn.cursor()
        c.execute("select distinct postPositivity from FacebookData where imageId = '{0}'".format(imageId))
        results = c.fetchall()
        c.close()
        if results:
            return results[0][0]

    def getShareCount(self, imageId):
        conn = db_location
        c = conn.cursor()
        c.execute("select distinct shareCount from FacebookData where imageId = '{0}'".format(imageId))
        results = c.fetchall()
        c.close()
        if results:
            return results[0][0]

    def setZeroCommentCountsToNegativeOne(self):
        conn = db_location
        c = conn.cursor()
        c.execute("UPDATE FacebookData SET commentCount = -1 WHERE commentCount < 1")
        conn.commit()
        c.close()

    def getMaxShareCount(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT max(shareCount) FROM FacebookData")
        max = c.fetchall()
        c.close()
        return max

    def getMaxCommentCount(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT max(commentCount) FROM FacebookData")
        max = c.fetchall()
        c.close()
        return max

    def getPositiveCommentCount(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT count(commentCount) FROM FacebookData WHERE commentCount > 0")
        results = c.fetchall()
        return results[0][0]

    def getImageIdWithPositiveCommentCounts(self):
        conn = db_location
        c = conn.cursor()
        c.execute(
            "SELECT DISTINCT imageId FROM FacebookData WHERE imageUrl LIKE '%http%' AND commentCount > 0 AND message != -1 AND postPositivity > -1")
        results = c.fetchall()
        c.close()
        return results

    def getFacebookDataWithPositiveCommentCounts(self):
        conn = db_location
        c = conn.cursor()
        c.execute(
            "SELECT DISTINCT * FROM FacebookData WHERE imageUrl LIKE '%http%' AND commentCount > 0 AND message != -1 AND postPositivity > -1")
        results = c.fetchall()
        c.close()
        return results

    def getCommentCount(self, imageId):
        conn = db_location
        c = conn.cursor()
        c.execute("select distinct commentCount from FacebookData where imageId = '{0}'".format(imageId))
        results = c.fetchall()
        c.close()
        if results:
            return results[0][0]

            #    def add_column(self):
            #        conn = db_location
            #        c = conn.cursor()
            #        c.execute("""ALTER TABLE FacebookData ADD COLUMN message TEXT DEFAULT '-1';""")
            #        conn.commit()
            #        c.close()

    def insertFacebookData(self, imageId, imageUrl, message, shareCount=None, commentCount=None, fanCount=None,
                           numberOfRatings=None, talkingAboutCount=None, pageRating=None, postPositivity=None,
                           postSubjectivity=None):
        conn = db_location
        c = conn.cursor()
        message = unicodedata.normalize('NFKD', message).encode('ascii', 'ignore').decode('ascii').replace("'", "''")
        # 0,2,3,4
        executeText = '''INSERT OR REPLACE INTO FacebookData(imageId, imageUrl, message, shareCount, commentCount, fanCount, numberOfRatings,talkingAboutCount,pageRating,postPositivity,postSubjectivity) 
                values('{0}','{1}','{2}','{3}','{4}','{5}','{6}','{7}','{8}','{9}','{10}');'''.format(
            imageId, imageUrl, message, shareCount, commentCount, fanCount, numberOfRatings, talkingAboutCount,
            pageRating, postPositivity, postSubjectivity)
        c.execute(executeText)
        conn.commit()
        c.close()

    def insertSentimentData(self, postPositivity, postSubjectivity, imageId):
        conn = db_location
        c = conn.cursor()
        c.execute(
            '''Update FacebookData set postPositivity = '{1}', postSubjectivity = '{2}' where imageId = '{0}';'''.format(
                imageId, postPositivity, postSubjectivity))
        conn.commit()
        c.close()

    def insertCommentCountData(self, commentCount, imageId):
        conn = db_location
        c = conn.cursor()
        c.execute(
            '''Update FacebookData set commentCount = '{1}' where imageId = '{0}';'''.format(
                imageId, commentCount))
        conn.commit()
        c.close()

    def getRow(self, imageId):
        conn = db_location
        c = conn.cursor()
        c.execute("select distinct * from FacebookData where imageId = '{0}'".format(imageId))
        results = c.fetchall()
        c.close()
        return results

    def selectColumnData(self, columnName):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT {0} FROM FacebookData".format(columnName))
        results = c.fetchall()
        c.close()
        return results

    def selectFacebookData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT * FROM FacebookData WHERE imageUrl LIKE '%http%'")
        results = c.fetchall()
        c.close()
        return results

    def selectCommentData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT commentCount, message FROM FacebookData WHERE imageUrl LIKE '%http%'")
        results = c.fetchall()
        c.close()
        return results

    def selectShareMessageData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT shareCount, message FROM FacebookData WHERE imageUrl LIKE '%http%'")
        results = c.fetchall()
        c.close()
        return results

    def selectSentimentMessageData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT postPositivity, message FROM FacebookData WHERE imageUrl LIKE '%http%'")
        results = c.fetchall()
        c.close()
        return results

    def selectFacebookDataWithNoPostPositivity(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT * FROM FacebookData WHERE imageUrl LIKE '%http%' AND postPositivity = -1")
        results = c.fetchall()
        c.close()
        return results

    def selectFacebookDataWithZeroCounts(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT * FROM FacebookData WHERE imageUrl LIKE '%http%' AND commentCount <= 0")
        results = c.fetchall()
        c.close()
        return results

    def getColumnNames(self):
        conn = db_location
        c = conn.cursor()
        c.execute("PRAGMA table_info(FacebookData);")
        results = c.fetchall()
        c.close()
        return results

    def getDataWithText(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT * FROM FacebookData WHERE message != -1")
        results = c.fetchall()
        c.close()
        return results

    def getPostText(self, imageId):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT DISTINCT * FROM FacebookData WHERE imageUrl LIKE '%http%'")
        results = c.fetchall()
        c.close()
        return results

    def deleteFacebookData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("DELETE FROM FacebookData WHERE 1 = 1")
        conn.commit()
        c.close()

    def isPageInDb(self, page_key):
        conn = db_location
        c = conn.cursor()
        c.execute("select imageId from FacebookData where imageId like '{0}_%' limit 1".format(page_key))
        results = c.fetchall()
        c.close()
        print(results)
        return len(results) > 0

    def dropFacebookTable(self):
        conn = db_location
        c = conn.cursor()
        c.execute("DROP TABLE FacebookData")
        conn.commit()
        c.close()  # facebookDb = FacebookDataDatabase()


if __name__ == '__main__':
    pass
    # facebookDb.add_column()
    # print(facebookDb.isPageInDb("174338385953299"))
