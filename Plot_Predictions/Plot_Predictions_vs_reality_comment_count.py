import random
import numpy
from matplotlib import pyplot

import os
import sys
from sklearn.model_selection import train_test_split

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

sys.path.append("../../")
from Notebooks.Text.Process import to_vector
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
from keras.models import load_model

print(os.getcwd())
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Static:
    facebookDb = FacebookDataDatabase()
    metric_getter = facebookDb.getCommentCount  # Set this to change the model type
    group_size = 10000
    limit = 10000
    plot_losses = PlotLearning("combined_keras_model")


def save_model(model):
    pkl_filename = Static.metric_getter.__name__ + ".pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def get_model():
    model_name = Static.metric_getter.__name__ + ".pkl"
    with open(model_name, 'rb') as pickle_file:
        return pickle.load(pickle_file)


def group(iterator, count):
    itr = iter(iterator)
    while True:
        yield tuple([next(itr) for i in range(count)])


def get_size(filename):
    st = os.stat(filename)
    return st.st_size


def to_array(images):
    return np.array(list(map(lambda x: img_to_array(x) / 255, images)))


def load_images(files):
    exists = list(filter(lambda x: os.path.exists(x), files))
    return list(map(lambda x: image.load_img(x, target_size=(64, 64)), exists))


def transform_image_data(all_files):
    files = list(filter(lambda x: ".jpg" in x or ".png" in x, filter(lambda x: ".DS_Store" not in x, all_files)))
    image_paths = list(map(lambda x: os.path.join("../Image_CNN/images", x), files))
    image_paths = list(filter(lambda x: get_size(x) > 1, image_paths))
    image_arrays = to_array(load_images(image_paths))
    return image_arrays


def get_data():
    all_files = os.listdir("../Image_CNN/images")[:Static.limit:]
    ids = list(map(lambda x: x[:-4], all_files))
    # for id in ids:
    # value = Static.facebookDb.getRow(id)
    # print(id, ": ", value)
    rows = list(
        map(lambda x: x[0], filter(lambda x: x if x else None, map(lambda x: Static.facebookDb.getRow(x), ids))))
    data = list(map(lambda x: (x[0], x[10], x[2], x[3]), rows))
    messages = list(map(lambda x: x[1], data))
    # share_counts = list(map(lambda x: x[2], data))
    comment_counts = list(map(lambda x: x[3], data))
    image_data = transform_image_data(all_files)
    message_data = to_vector(messages)
    y_data = list(map(lambda x: x if x > 0 else 0, map(lambda x: Static.metric_getter(x), ids)))
    combined_data = zip(image_data, message_data, y_data)
    import statistics
    print("Data Var: ", statistics.stdev(y_data) ** 2)
    image_data_batch, message_data_batch, y_data_batch = zip(*combined_data)
    (trainX_image, testX_image, trainY_image, testY_image) = train_test_split(image_data_batch, y_data_batch,
                                                                              test_size=0.25, random_state=42)
    (trainX_message, testX_message, trainY_message, testY_message) = train_test_split(message_data_batch,
                                                                                      y_data_batch, test_size=0.25,
                                                                                      random_state=42)
    assert trainY_image == trainY_message
    assert testY_image == testY_message
    return trainX_image, trainX_message, trainY_image, testX_image, testX_message, testY_image


from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def get_predictions(image_data, message_data, answers, image_model, message_model):
    message_data_squeezed = np.squeeze(np.array(message_data), axis=1)
    image_predictions = image_model.predict(np.array(image_data)).tolist()
    message_predictions = message_model.predict(message_data_squeezed).tolist()
    predictions = [[i[0], j[0]] for i, j in zip(image_predictions, message_predictions)]
    return predictions


trainX_image, trainX_message, trainY, testX_image, testX_message, testY = get_data()
model = load_model(
    "/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Combine_Keras_Models/Combined_Model_getCommentCount.h5")
predictions = model.predict([np.array(trainX_image), np.squeeze(np.array(trainX_message), axis=1)])
flatten_predictions = list(map(lambda x: x[0], predictions))

bins = numpy.linspace(0, 200, 75)

pyplot.hist(flatten_predictions, bins, alpha=0.5, label='Comment Count Prediction')
pyplot.hist(testY, bins, alpha=0.5, label='Comment Count', )
pyplot.title("Histogram of Prediction Comment Count\non Actual Comment Count Histogram")
pyplot.legend(loc='upper right')
pyplot.xlabel("Comment Count")
pyplot.ylabel("Bin Count")
pyplot.savefig('CommentCountPredictionHistogram.png')
