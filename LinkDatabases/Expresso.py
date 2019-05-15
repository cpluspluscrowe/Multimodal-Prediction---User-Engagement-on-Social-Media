import os
import sqlite3

file_location = os.path.dirname(os.path.realpath(__file__))
db_location = sqlite3.connect('{0}/AdData.db'.format(file_location))


class ExpressoLinkDatabase():
    def __init__(self):
        conn = db_location
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS ExpressoLink (link TEXT);')
        conn.commit()
        c.close()

    def insertExpressoLinks(self, links):
        for link in links:
            self.insertLink(link)

    def insertLink(self, link):
        if type(link) == list:
            self.insertExpressoLinks(link)
        else:
            conn = db_location
            c = conn.cursor()
            c.execute('CREATE TABLE IF NOT EXISTS ExpressoLink (link TEXT);')
            c.execute('INSERT OR REPLACE INTO ExpressoLink(link) values("{0}");'.format(link))
            conn.commit()
            c.close()

    def selectExpressoData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT link FROM ExpressoLink WHERE link LIKE '%http%'")
        results = c.fetchall()
        c.close()
        return results

    def deleteExpressoData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("DELETE FROM ExpressoLink WHERE 1 = 1")
        conn.commit()
        c.close()


expressoDb = ExpressoLinkDatabase()
# expressoDb.insertLink("ahttp://www.google.com")
# expressoDb.deleteExpressoData()
