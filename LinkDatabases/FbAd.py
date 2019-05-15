import os
import sqlite3

file_location = os.path.dirname(os.path.realpath(__file__))
db_location = sqlite3.connect('{0}/AdData.db'.format(file_location))


class AdLinkDatabase():
    def __init__(self):
        conn = db_location
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS AdLink (espressoPage TEXT,link TEXT);')
        conn.commit()
        c.close()

    def insertAdLinks(self, links, espressoPageLink):
        if type(links) == str:  # check edge case
            self.insertLink(links, espressoPageLink)
        else:
            for link in links:
                self.insertLink(link, espressoPageLink)

    def insertLink(self, link, espressoPageLink):
        if type(link) == list:  # check edge case
            self.insertAdLinks(link, espressoPageLink)
        else:
            conn = db_location
            c = conn.cursor()
            c.execute(
                'INSERT OR REPLACE INTO AdLink(espressoPage, link) values("{0}","{1}");'.format(espressoPageLink, link))
            conn.commit()
            c.close()

    def deleteAdData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("DELETE FROM AdLink WHERE 1 = 1")
        conn.commit()
        c.close()

    def hasEspressoUrl(self, espressoUrl):
        conn = db_location
        c = conn.cursor()
        c.execute('select * from AdLink where espressoPage like "{0}"'.format(espressoUrl))
        results = c.fetchall()
        c.close()
        return len(results) > 0

    def selectAdData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("SELECT * FROM AdLink WHERE link LIKE '%http%'")
        results = c.fetchall()
        c.close()
        return results

    def dropAdLinkTable(self):
        conn = db_location
        c = conn.cursor()
        c.execute("DROP TABLE AdLink")
        conn.commit()
        c.close()


adDb = AdLinkDatabase()
# adDb.insertLink("http:fake")
# adDb.deleteAdData()
