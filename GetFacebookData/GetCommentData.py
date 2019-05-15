# coding: utf-8

# In[1]:


import os
import sys
from datetime import datetime

sys.path.append("../../")

from Notebooks.SearchFbData.GetKeyData import get_key_data
from Notebooks.Token.GenerateToken import getToken
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
from Notebooks.LinkDatabases.PostComments import PostDataDatabase
import requests

import ast
import unicodedata

facebookDb = FacebookDataDatabase()
commentDb = PostDataDatabase()
global token
token = getToken()


# Get/Set post data.  These are person posts on the Ad in question
class Comment:
    def __init__(self):
        self.id = None
        self.message = None
        self.like_count = None

    def setId(self, id):
        self.id = id

    def setMessage(self, message):
        self.message = unicodedata.normalize('NFKD', message).encode('ascii', 'ignore').decode('ascii').replace("'",
                                                                                                                "''")

    def setLikeCount(self, like_count):
        self.like_count = like_count


def get_post_on_comment_url(comment):
    if comment.id:
        global token
        token = getToken()
        url = '''https://graph.facebook.com/v2.11/{0}?access_token={1}&debug=all&format=json&method=get&pretty=0&suppress_http_code=1000'''.format(
            comment.id, token)
        post = requests.get(url)
        raw = ast.literal_eval(post.text)
        get_key_data(raw, "message", comment.setMessage)  # set the message for the comment


def writeIndex(index):
    with open("./index.txt", "w") as f:
        f.write(str(index))


def readIndex():
    with open("./index.txt", "r") as f:
        return int(f.read())


def storeOutput():
    pass


index = readIndex()
data = facebookDb.selectFacebookData()


def fill_queue():
    for cnt in range(index, index + 100000):
        q.put(cnt)
        print(q.qsize())


class InsertData:
    def __init__(self, comment_count, post_id):
        self.comment_count = comment_count
        self.post_id = post_id


from queue import Queue

to_insert = Queue()

import sys


def run():
    index = q.get()
    post = data[index]
    global token
    token = getToken()
    post_id = post[0]
    comment_url = '''https://graph.facebook.com/v2.11/{0}?fields=comments.limit(1000000)&access_token={1}&debug=all&format=json&method=get&pretty=0&suppress_http_code=1000'''.format(
        post_id, token)
    post_data = requests.get(comment_url)
    try:
        post_raw_data = ast.literal_eval(post_data.text)
    except:
        sys.exit(post_data.text)
    if "comments" in post_raw_data:
        comment_count = len(post_raw_data["comments"]["data"])
        sys.stdout.write("Count: {0}\n".format(comment_count))
        to_insert.put(InsertData(comment_count, post_id))
    else:
        to_insert.put(InsertData(0, post_id))
        sys.stdout.write("Storing index {0}\n".format(index))
    writeIndex(index)


from threading import Thread

q = Queue()
fill_queue()

while not q.empty():
    print("Queue Size: ", q.qsize())
    print("Number of positive comments: ", facebookDb.getPositiveCommentCount())
    threads = []
    for thread in range(100):
        t1 = Thread(target=run)
        t1.start()
        threads.append(t1)

    for thread in threads:
        thread.join()

while not to_insert.empty():
    data = to_insert.get()
    comment_count = data.comment_count
    post_id = data.post_id
    facebookDb.insertCommentCountData(comment_count, post_id)

# posts = facebookDb.selectFacebookData()
# tgc = ThreadCommentGetter()
# tgc.run()
import time

print("Done", datetime.now())
