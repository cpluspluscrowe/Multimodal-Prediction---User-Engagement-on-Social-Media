import os
import sys
from sklearn.model_selection import train_test_split

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

sys.path.append(os.pardir)
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
from itertools import combinations
from scipy.stats import chi2_contingency

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Static:
    facebookDb = FacebookDataDatabase()
    metric_getter = facebookDb.getShareCount  # Set this to change the model type
    limit = len(os.listdir("../Image_CNN/images"))
    group_size = limit
    plot_losses = PlotLearning("combined_keras_model")
    batch_size = 1
    epochs = 20


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
    loaded = []
    for file_to_load in files:
        try:
            load = image.load_img(file_to_load, target_size=(64, 64))
            loaded.append(load)
        except:
            print("Could not load file: ", file_to_load, os.path.exists(file_to_load))
    return loaded


def transform_image_data(all_files):
    files = list(filter(lambda x: ".jpg" in x or ".png" in x, filter(lambda x: ".DS_Store" not in x, all_files)))
    image_paths = list(map(lambda x: os.path.join("../Image_CNN/images", x), files))
    image_paths = list(filter(lambda x: get_size(x) > 1, image_paths))
    image_arrays = to_array(load_images(image_paths))
    return image_arrays


def get_data():
    sentiment_post_ids = list(map(lambda x: x[0], Static.facebookDb.getImageIdWithPositiveCommentCounts()))
    all_files = os.listdir("../Image_CNN/images")[:Static.limit]

    print("Number of files in analysis: {0}".format(len(all_files)))
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
    y_data = list(map(lambda x: x if x > 0 else 0,
                      filter(lambda x: x != None, map(lambda x: Static.metric_getter(x), ids))))
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


def find_models(model_name):
    model_paths = []
    for file in os.listdir("./"):
        if model_name in file and ".h5" in file:
            model_paths.append(os.path.join("./", file))
    return model_paths


def get_models():
    loaded_model = []  # print(get_models())
    for model_path in find_models(Static.metric_getter.__name__):
        model = load_model(model_path)
        loaded_model.append(model)
    return loaded_model


def get_model_by_name(name):
    model = load_model(find_models(name)[0])
    return model


models = get_models()

from sklearn.cross_validation import train_test_split


def get_predictions(image_data, message_data, answers, image_model, message_model):
    message_data_squeezed = np.squeeze(np.array(message_data), axis=1)
    image_predictions = image_model.predict(np.array(image_data)).tolist()
    message_predictions = message_model.predict(message_data_squeezed).tolist()
    predictions = [[i[0], j[0]] for i, j in zip(image_predictions, message_predictions)]
    return predictions


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


newModel = load_model(
    r"/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Combine_Keras_Models/Combined_Model_getSentiment.h5")
trainX_image, trainX_message, trainY, testX_image, testX_message, testY = get_data()
trainY = list(map(lambda x: x * 100, trainY))
testY = list(map(lambda x: x * 100, testY))

nested_predictions = newModel.predict([np.array(trainX_image), np.squeeze(np.array(trainX_message), axis=1)])
predictions = list(map(lambda x: x[0], nested_predictions))
truths = testY
incorrect = 0
correct = 0
for post1_tuple, post2_tuple in combinations(list(zip(truths, predictions)), 2):
    post1 = PostPrediction(post1_tuple)
    post2 = PostPrediction(post2_tuple)
    if abs(post1.prediction - post2.prediction) > 33:
        truth1 = post1.prediction > post2.prediction
        truth2 = post1.truth > post2.truth
        print(post1.truth, post1.prediction)
        print(post2.truth, post2.prediction)
        if incorrect > 0 or correct > 0:
            print(correct, incorrect, correct / (incorrect + correct))
        print()
        if truth1 != truth2:
            incorrect += 1
        else:
            correct += 1
print("Correct: ", correct, " Incorrect: ", incorrect, " Percent Correct: ", correct / (correct + incorrect),
      chi_prob(correct, incorrect))
