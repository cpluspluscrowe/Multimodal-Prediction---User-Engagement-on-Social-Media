import os
import sys
sys.path.append(os.pardir)
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
import os
facebookDb = FacebookDataDatabase()
files = os.listdir("./images")

delete_paths = []

ids = list(map(lambda x: x.replace(".png","").replace(".jpg",""),files))
for x in ids:
    shareCount = facebookDb.getShareCount(x)
    if shareCount is None:
        print(x)
        delete_paths.append(os.path.join("./images",x))

print()
for x in ids:
    shareCount = facebookDb.getCommentCount(x)
    if shareCount is None:
        print(x)
        delete_paths.append(os.path.join("./images",x))

#should_delete = input("Should we delete these files?")
#if should_delete == 'y' or should_delete == 'yes':
#    for path in delete_paths:
#        os.remove(path)

