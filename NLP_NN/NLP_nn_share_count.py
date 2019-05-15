import os
import sys
from itertools import combinations

from keras import regularizers
from scipy.stats import chi2_contingency

sys.path.append(os.pardir)
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase

import pandas as pd
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split
from Notebooks.Text.Process import PostText, to_vector
from Notebooks.Mapper.Message import MessageGetter
from Notebooks.Plot.plot_training import PlotLearning
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords


class Global:
    batch_size = 256
    epochs = 100
    group_size = 100_000
    plot_losses = PlotLearning("nlp_nn_share_count")
    facebookDb = FacebookDataDatabase()
    regularizer_function = None


postData = MessageGetter.facebookDb.selectShareMessageData()
df = pd.DataFrame(postData)
X_train, X_test, y_train, y_test = train_test_split(df[1], df[0], test_size=0.1)

vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=4000)

X_train_onehot = vectorizer.fit_transform(X_train)
X_test_onehot = vectorizer.fit_transform(X_test)

model = Sequential()

model.add(Dense(units=512, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dropout(0.2))
model.add(Dense(units=256, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dropout(0.2))
model.add(Dense(units=128, activation='relu', input_dim=len(vectorizer.get_feature_names())))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='relu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(X_train_onehot[:-20000], y_train[:-20000],
          epochs=10, batch_size=1024, verbose=1,
          validation_data=(X_train_onehot[-20_000:], y_train[-20_000:]), callbacks=[Global.plot_losses])

scores = model.evaluate(vectorizer.transform(X_test), y_test, verbose=1)
print(scores)
print("Accuracy:", scores[1])  # Accuracy: 0.875

model_name = '/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/NLP_NN/nlp_getShareCount.h5'
model.save(model_name)


class PostPrediction:
    def __init__(self, truth_prediction_tuple):
        self.truth = truth_prediction_tuple[0]
        self.prediction = truth_prediction_tuple[1]


def chi_prob(right, wrong):
    expected = (right + wrong) / 2
    observed = np.array([[right, wrong], [expected, expected]])
    chi2, p, dof, expected = chi2_contingency(observed)
    if right < wrong:
        print("Incorrect more often than correct.  Not statistically significant")
    print("Chi: ", chi2, "p-value: ", p, "dof: ", dof)


nested_predictions = model.predict(vectorizer.transform(X_test))
predictions = list(map(lambda x: x[0], nested_predictions))
truths = y_test
incorrect = 0
correct = 0
cnt = 0
for post1_tuple, post2_tuple in combinations(list(zip(truths, predictions)), 2):
    post1 = PostPrediction(post1_tuple)
    post2 = PostPrediction(post2_tuple)
    truth1 = post1.prediction > post2.prediction
    truth2 = post1.truth > post2.truth
    if truth1 != truth2:
        incorrect += 1
    else:
        correct += 1
    cnt += 1
    if cnt % 10000 == 0:
        print("Correct: ", correct, " Incorrect: ", incorrect, " Percent Correct: ", correct / (correct + incorrect),
              chi_prob(correct, incorrect))

    if cnt > 10000000:
        exit()
print("Correct: ", correct, " Incorrect: ", incorrect, " Percent Correct: ", correct / (correct + incorrect),
      chi_prob(correct, incorrect))
