import os
import sqlite3

file_location = os.path.dirname(os.path.realpath(__file__))
db_location = sqlite3.connect('{0}/Features.db'.format(file_location))


class DescriptorsDatabase():
    def __init__(self):
        conn = db_location
        c = conn.cursor()
        c.execute('CREATE TABLE IF NOT EXISTS Descriptors (id INT PRIMARY KEY, descriptor BLOB);')
        conn.commit()
        c.close()

    def insertDescriptorData(self, id, descriptor):
        conn = db_location
        c = conn.cursor()
        c.execute('INSERT OR REPLACE  INTO Descriptors(id, descriptor) values("{0}","{1}");'.format(id, descriptor))
        conn.commit()
        c.close()

    def getDescriptor(self, id):
        conn = db_location
        c = conn.cursor()
        c.execute("""select descriptor from Descriptors where id = '{0}'""".format(id))
        results = c.fetchall()
        c.close()
        return results

    def selectAll(self):
        conn = db_location
        c = conn.cursor()
        c.execute("""SELECT * FROM Descriptors LIMIT 100""")
        results = c.fetchall()
        c.close()
        return results

    def deleteDescriptorsData(self):
        conn = db_location
        c = conn.cursor()
        c.execute("DELETE FROM Descriptors WHERE 1 = 1")
        conn.commit()
        c.close()


if __name__ == "__main__":
    descriptorsDb = DescriptorsDatabase()
    # descriptorsDb.deleteDescriptorsData()
    # descriptorsDb.insertDescriptorData(1,"some data")
    print(descriptorsDb.getDescriptor(1))
