# Setup Facebook API graph object
import os
import sys

sys.path.append(os.pardir)
from Notebooks.LinkDatabases.FbAd import adDb
from Notebooks.Token.GenerateToken import getToken
from Notebooks.Metrics.GetFacebookMetrics import getPageMetrics
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
from Notebooks.SearchFbData.GetKeyData import escape_url, get_key_data
import requests
import ast

facebookDb = FacebookDataDatabase()


class Page:
    def __init__(self):
        self.posts = []
        self.followers_count = None
        self.likes_count = None
        self.reigon = None
        self.page_rating = None
        self.review_count = None
        self.metrics = None


class Post:
    def __init__(self):
        self.message = None
        self.image_url = None
        self.id = None
        self.image_url = None
        self.share_count = 0
        self.comment_count = 0
        self.comments = []

    def setShareCount(self, share_count):
        self.share_count = share_count

    def setImageUrl(self, image_url):
        self.image_url = image_url


def modifyUrl(url):
    url = url[0]
    id1 = url[url.find(start_url_string) + len(start_url_string):url.find(r"/posts")]
    page_url = graph_url_start + id1 + end_token
    return page_url


def getPostData(url):
    main_page = requests.get(url)
    page_raw = ast.literal_eval(main_page.text)
    if "error" in page_raw:
        url = url.replace(".limit(1000)", ".limit(100)")
        if "message" in page_raw:
            if "reduce the amount of data" in page_raw["message"]:
                print("Reducing")
                main_page = requests.get(url)
                page_raw = ast.literal_eval(main_page.text)
            else:
                print("Error", page_raw, url)
                token = getToken()
                return getPostData(url)
        else:
            return
    page_data = page_raw["posts"]["data"]
    posts = []
    for post_raw in page_data:
        post = Post()
        if "message" in post_raw:
            post.message = escape_url(post_raw["message"])
            post.id = post_raw["id"]
            posts.append(post)
    return posts


# Main Page Posts: Get Message and the number of shares

def get_attachment_data(post_id):
    url = '''https://graph.facebook.com/v2.11/{0}/attachments?access_token={1}&debug=all&format=json&method=get&pretty=0&suppress_http_code=1000'''.format(
        post_id, token)
    post = requests.get(url)
    post_raw = ast.literal_eval(post.text)
    return post_raw


def get_images(post):
    post_raw = get_attachment_data(post.id)
    get_key_data(post_raw, "src", post.setImageUrl)


def get_and_set_share_count(comment):
    url = '''https://graph.facebook.com/v2.11/{0}?fields=shares&access_token={1}&debug=all&format=json&method=get&pretty=0&suppress_http_code=1000'''.format(
        comment.id, token)
    share_data = requests.get(url)
    data = ast.literal_eval(share_data.text)
    get_key_data(data, "count", comment.setShareCount)


def StoreInFacebookData(imageId, imageUrl, message, shareCount, commentCount, fanCount, numberOfRatings,
                        talkingAboutCount, pageRating, commentPositivity):
    if imageUrl:
        facebookDb.insertFacebookData(imageId, imageUrl, message, shareCount, commentCount, fanCount, numberOfRatings,
                                      talkingAboutCount, pageRating, commentPositivity)
        # if shareCount > 0 or commentCount > 0:
        #    print(imageId, imageUrl,shareCount,commentCount)


# get the image and share count for the post, store this and the page metrics in the facebook database
def set_post_data(post):
    get_images(post)
    get_and_set_share_count(post)


def setGlobals():
    global token
    global start_url_string
    global graph_url_start
    global end_token
    token = getToken()
    start_url_string = r"https://facebook.com/"
    graph_url_start = r"https://graph.facebook.com/v2.11/"
    end_token = '''?access_token={0}&fields=posts.limit(1000)&debug=all&format=json&method=get&pretty=0&suppress_http_code=1000'''.format(
        token)


setGlobals()


def get_page_id(url):
    return url[url.find("v2.11/") + 6:url.find("?")]


def run():
    indexStart = 0
    facebookDb = FacebookDataDatabase()
    for raw_url in adDb.selectAdData()[indexStart:]:
        global token
        token = getToken()
        url = modifyUrl(raw_url)
        post_number = get_page_id(url)
        if not facebookDb.isPageInDb(post_number):
            page = Page()
            page.metrics = getPageMetrics(url)
            page.posts = getPostData(url)
            if page.posts:
                for post in page.posts:
                    token = getToken()
                    set_post_data(post)
                    StoreInFacebookData(post.id, post.image_url, post.message, post.share_count, post.comment_count,
                                        page.metrics.fan_count, page.metrics.rating_count,
                                        page.metrics.talking_about_count, page.metrics.star_rating, -1)
                    print("Stored!", post.id)
        else:
            pass


import time
while True:
    try:
        run()
    except Exception as e:
        print(e.message)
        from datetime import datetime
        print(datetime.now())
        print("Error in thread")
        time.sleep(1000)

# tgc.run()



print(len(facebookDb.selectFacebookData()))
