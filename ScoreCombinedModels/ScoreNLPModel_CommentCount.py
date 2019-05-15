import os
import sys

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

sys.path.append(os.pardir)
import pandas as pd
import pickle
from PIL import ImageFile
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.image import img_to_array
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten
from keras.models import load_model
from keras.layers import *
from keras.models import Model
import keras
from Notebooks.Plot.plot_training import PlotLearning
from itertools import combinations
from scipy.stats import chi2_contingency

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Static:
    facebookDb = FacebookDataDatabase()
    metric_getter = facebookDb.getCommentCount  # Set this to change the model type
    limit = 30000


postData = Static.facebookDb.selectCommentData()[:Static.limit]
df = pd.DataFrame(postData)
X_train, X_test, y_train, y_test = train_test_split(df[1], df[0], test_size=0.2)

newModel = load_model(
    r"/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/NLP_NN/nlp_getCommentCount.h5")

vectorizer = CountVectorizer(binary=True, stop_words=stopwords.words('english'),
                             lowercase=True, min_df=3, max_df=0.9, max_features=4000)

X_train_onehot = vectorizer.fit_transform(X_train)
X_test_onehot = vectorizer.fit_transform(X_test)
