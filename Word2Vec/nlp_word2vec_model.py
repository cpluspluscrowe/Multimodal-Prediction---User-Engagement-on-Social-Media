import os
import sys

from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
import numpy as np

sys.path.append(os.pardir)
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
import time

facebookDb = FacebookDataDatabase()
import pandas as pd
from Notebooks.Text.Process import PostText
from Notebooks.Mapper.Message_Sentiment import MessageGetter
from Notebooks.Plot.plot_training import PlotLearning
import pickle


def store_df(dataframe):
    path = "df.obj"
    filehandler = open(path, "wb")
    pickle.dump(dataframe, filehandler)
    filehandler.close()


def get_df():
    path = "df.obj"
    if os.path.exists(path):
        file = open(path, 'rb')
        dataframe = pickle.load(file)
        file.close()
        return dataframe


class Global:
    batch_size = 16
    epochs = 5
    group_size = None
    plot_losses = PlotLearning("nlp_nn")


def train_word2vec_model(sentences, vector_dim):
    model = Word2Vec(sentences, size=vector_dim, window=5, min_count=1, workers=1, hs=1)
    return model


def generate_df():
    postTexts = []
    for post in MessageGetter.get_post_generator():
        pt = PostText(post)
        postTexts.append(pt.getValues())
    df = pd.DataFrame.from_records(postTexts, columns=MessageGetter.get_columns())[
        ['message', 'postPositivity']]
    df = df[(df['postPositivity'] > -1)]
    return df


def split_message(message):
    split_sentence = message.split()
    return split_sentence


def train_model(df, vector_dim):
    df["message"] = df["message"].apply(split_message)

    trainingSet, testSet = train_test_split(df, test_size=0.2)
    training_setences = list(trainingSet["message"])
    model = train_word2vec_model(training_setences, vector_dim)
    print(model)
    return model, df


def list_mean(probabilities):
    if sum(probabilities) == 0:
        return 0
    probabilities = [x for x in probabilities if x != 0.0]
    probabilities = list(map(lambda x: 2 ** x, probabilities))
    return float(sum(probabilities)) / max(len(probabilities), 1)


def get_score(model, text):
    try:
        return list_mean(model.score(text))
    except:
        return -2


def score_models(correct_model, wrong_model1, wrong_model2, test_set):
    right_count = 0
    wrong_count = 0
    for text in list(test_set["message"]):
        text = list(text)
        right_score = get_score(correct_model, text)
        wrong1_score = get_score(wrong_model1, text)
        wrong2_score = get_score(wrong_model2, text)
        if right_score > -2:
            if right_score > wrong1_score and right_score > wrong2_score:
                right_count += 1
            else:
                wrong_count += 1
    if (wrong_count + right_count > 0):
        return right_count / (wrong_count + right_count)
    else:
        return 0


def run_models(vector_size):
    print("Training Models")

    highly_positive_df = df[df['postPositivity'] >= 0.7]
    slightly_positive_df = df[(df.postPositivity < 0.7) & (df.postPositivity > 0.3)]
    negative_df = df[(df['postPositivity'] <= 0.3) & (df['postPositivity'] > -1)]
    print("Data Count in each model: Very Positive : {0}; Slightly: {1}; Negative: {2}"
          .format(len(highly_positive_df),
                  len(
                      slightly_positive_df),
                  len(negative_df)))
    print("Vector Dim: {0}".format(vector_size))

    highly_positive_model, hp_test = train_model(highly_positive_df, vector_size)
    print("Finished training positive model")

    slightly_positive_model, sp_test = train_model(slightly_positive_df, vector_size)
    print("finished training slightly positive model")

    negative_model, negative_test = train_model(negative_df, vector_size)
    print("finished training negative model")

    print("Positive: ", score_models(highly_positive_model, slightly_positive_model, negative_model, hp_test))
    print("Neutral: ", score_models(slightly_positive_model, highly_positive_model, negative_model, sp_test))
    print("Negative: ", score_models(negative_model, slightly_positive_model, highly_positive_model, negative_test))

    highly_positive_model.save("highly_positive.h5")
    slightly_positive_model.save("slightly_positive.h5")
    negative_model.save("negative_positive.h5")
    print()


from threading import Thread

df = None  # get_df()
if df is None:
    df = generate_df()
    store_df(df)

run_models(100)
df = get_df()
