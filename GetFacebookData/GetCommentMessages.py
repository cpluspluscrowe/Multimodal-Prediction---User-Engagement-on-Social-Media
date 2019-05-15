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
print("Number of distinct Posts: ", commentDb.getDistinctPostCount())
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


index = readIndex()
data = facebookDb.selectFacebookDataWithNoPostPositivity()
print("Length of data: {0}".format(len(data)))


def fill_queue():
    for cnt in range(index, index + 20000):
        q.put(cnt)
        print(q.qsize())


class InsertData:
    def __init__(self, comment_message, comment_id, comment_parent):
        self.message = comment_message
        self.comment_id = comment_id
        self.post_id = comment_parent


from queue import Queue

to_insert = Queue()

import sys

import time


def run(post_id=None):
    index = q.get()
    post = data[index]
    global token
    token = getToken()
    if not post_id:
        post_id = post[0]
    comment_url = '''https://graph.facebook.com/v2.11/{0}/comments?access_token={1}&debug=all&format=json&method=get&pretty=0&suppress_http_code=1000'''.format(
        post_id, token)
    post_data = requests.get(comment_url)
    try:
        post_raw_data = ast.literal_eval(post_data.text)
    except Exception as e:
        print(str(e))
        print(comment_url)
        sys.exit(post_data.text)
        # time.sleep(1000)
        # run(post_id)

    if "data" in post_raw_data:
        for comment_raw in post_raw_data["data"]:
            to_insert.put(InsertData(comment_raw["message"], comment_raw["id"], post_id))
    writeIndex(index)


from threading import Thread

q = Queue()
fill_queue()
cnt = 0
while not q.empty():
    print("Queue Size: ", q.qsize())
    threads = []
    for thread in range(20):
        t1 = Thread(target=run)
        t1.start()
        threads.append(t1)

    for thread in threads:
        thread.join()

    while not to_insert.empty():
        cnt += 1
        data_to_insert = to_insert.get()
        comment_message = unicodedata.normalize('NFKD', data_to_insert.message).encode('ascii', 'ignore').decode(
            'ascii').replace(
            "'",
            "''")
        comment_id = data_to_insert.comment_id
        comment_parent = data_to_insert.post_id
        commentDb.insertPostData(comment_parent, comment_id, comment_message)
    if cnt > 1000:
        print("Number of distinct Posts: ", commentDb.getDistinctPostCount())
        cnt = 0

print("Number of distinct Posts: ", commentDb.getDistinctPostCount())
print("Done", datetime.now())
