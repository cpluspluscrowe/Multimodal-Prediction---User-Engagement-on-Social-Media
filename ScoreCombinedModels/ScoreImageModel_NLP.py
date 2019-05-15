import itertools
import os
import sys

import numpy as np
import pandas as pd
from PIL import ImageFile
from keras.models import load_model
from scipy.stats import chi2_contingency

from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
from Notebooks.Mapper.Message import MessageGetter
from Notebooks.Text.Process import PostText, to_vector

sys.path.append(os.pardir)

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

facebookDb = FacebookDataDatabase()
metricGetter = facebookDb.getShareCount  # getCommentCount#
model_name = 'nlp_{0}.h5'.format(metricGetter.__name__)
if os.path.exists(model_name):
    print("Loading model")
    model = load_model(model_name)
    print("Loaded")
else:
    raise Exception("Model {0} does not exist".format(model_name))


def dict_factory(row, columns):
    d = {}
    for idx, col in enumerate(columns):
        d[col] = row[idx]
    return d


def get_columns():
    columns = facebookDb.getColumnNames()
    columns = list(map(lambda x: x[1], columns))
    return columns


def get_post():
    postData = facebookDb.selectFacebookData()
    for post in postData:
        post_obj = dict_factory(post, get_columns())
        yield post_obj


# loss = model.evaluate(x=testX, y=testY,batch_size=Global.batch_size)
from collections import defaultdict


class Post:
    def getShareCount(self):
        return self.metric

    def getCommentCount(self):
        return self.metric


class Page:
    # static
    files = os.listdir("./images")
    pages = defaultdict(lambda: [])
    vectors = {}

    def __init__(self):
        self.posts = []

    @staticmethod
    def getShareCount(post_obj):
        return post_obj["shareCount"]

    @staticmethod
    def getCommentCount(post_obj):
        return post_obj["commentCount"]

    @staticmethod
    def textTransform(post_obj, metricGetter):
        post = Post()
        post_id = post_obj["imageId"]
        post.metric = getattr(Page, metricGetter.__name__)(post_obj)
        post.image = Page.vectors[post_id]
        return post


def fill_pages_data():
    messages = []
    postTexts = []
    for post_obj in get_post():
        pt = PostText(post_obj)
        postTexts.append(pt.getValues())
    df = pd.DataFrame.from_records(postTexts, columns=MessageGetter.get_columns())
    vectors = to_vector(df["message"])

    for post_obj, vector in zip(get_post(), vectors):
        post_id = post_obj["imageId"]
        messages.append(post_obj["message"])
        page_id = post_obj["imageId"].split("_")[0]
        Page.pages[page_id].append(post_obj)
        Page.vectors[post_id] = vector


fill_pages_data()

sorted_by_number_of_posts = sorted(Page.pages.items(), key=lambda value: len(value[1]))


def chi_prob(right, wrong):
    expected = (right + wrong) / 2

    observed = np.array([[right, wrong], [expected, expected]])
    chi2, p, dof, expected = chi2_contingency(observed)
    print("Chi: ", chi2, "p-value: ", p, "dof: ", dof)


total = 0
neutral = 0
right = 0
wrong = 0
for entry in sorted_by_number_of_posts:
    page = entry[1]
    print("Number of Posts: ", len(page))

    page_objs = list(filter(lambda x: not x.image is None and not x.metric is None,
                            map(lambda x: Page.textTransform(x, metricGetter), page)))
    if len(page_objs) == 0:
        print("Skipping")
        continue
    if "Share" in metricGetter.__name__:
        name = "shareCount"
    else:
        name = "commentCount"
    truths = list(map(lambda x: x[name], page))
    text_vectors = np.array(list(map(lambda x: x.image, page_objs)))
    squeezed_text_vectors = np.squeeze(text_vectors, axis=1)
    predictions = model.predict(squeezed_text_vectors)
    valueCombos = list(zip(truths, predictions))
    for p1, p2 in itertools.combinations(valueCombos, 2):
        pt1, pp1 = p1
        pt2, pp2 = p2
        total += 1
        if pt1 == pt2 and pp1 == pp2:
            neutral += 1
            continue
        if (pt1 > pt2 and not pp1 > pp2) or (pt1 < pt2 and not pp1 < pp2):
            wrong += 1
        if (pt1 > pt2 and pp1 > pp2) or (pt1 < pt2 and pp1 < pp2):
            right += 1
        if wrong > 0:
            score = right / wrong
            print("Score: {0}".format(score))
            print("right:", right, "wrong", wrong)
            chi_prob(right, wrong)
            print(pt1, pp1, " ", pt2, pp2, "  ", score)
        else:
            print("Wrong is 0", wrong, right, neutral)
        print("total", total, "p1", pt1, pp1, "p2", pt2, pp2)
        import time

        time.sleep(1)
