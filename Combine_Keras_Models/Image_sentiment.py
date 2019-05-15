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

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Static:
    facebookDb = FacebookDataDatabase()
    metric_getter = facebookDb.getSentiment  # Set this to change the model type
    group_size = 25000
    limit = 30000
    plot_losses = PlotLearning("combined_keras_model_sentiment")
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
    all_dir_files = os.listdir("../Image_CNN/images")

    all_files = []
    for post_id in sentiment_post_ids:
        for file_name in all_dir_files:
            if post_id in file_name:
                all_files.append(file_name)
    print("Number of files in analysis: {0}".format(len(all_files)))
    print(all_files)
    ids = list(map(lambda x: x[:-4], all_files))
    rows = list(
        map(lambda x: x[0], filter(lambda x: x if x else None, map(lambda x: Static.facebookDb.getRow(x), ids))))
    data = list(map(lambda x: (x[0], x[10], x[2], x[3]), rows))
    messages = list(map(lambda x: x[1], data))
    image_data = transform_image_data(all_files)
    message_data = to_vector(messages)
    y_data = list(map(lambda x: x if x > 0 else 0,
                      filter(lambda x: x != None, map(lambda x: Static.metric_getter(x), ids))))
    combined_data = zip(image_data, message_data, y_data)
    if Static.group_size is None:
        Static.group_size = len(list(combined_data)) - 10000
    import statistics
    print("Data Var: ", statistics.stdev(y_data) ** 2)
    for data_chunk in group(combined_data, Static.group_size):  # ):group(
        image_data_batch, message_data_batch, y_data_batch = zip(*data_chunk)
        (trainX_image, testX_image, trainY_image, testY_image) = train_test_split(image_data_batch, y_data_batch,
                                                                                  test_size=0.25, random_state=42)
        (trainX_message, testX_message, trainY_message, testY_message) = train_test_split(message_data_batch,
                                                                                          y_data_batch, test_size=0.25,
                                                                                          random_state=42)
        assert trainY_image == trainY_message
        assert testY_image == testY_message
        yield trainX_image, trainX_message, trainY_image, testX_image, testX_message, testY_image


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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def get_predictions(image_data, message_data, answers, image_model, message_model):
    message_data_squeezed = np.squeeze(np.array(message_data), axis=1)
    image_predictions = image_model.predict(np.array(image_data)).tolist()
    message_predictions = message_model.predict(message_data_squeezed).tolist()
    predictions = [[i[0], j[0]] for i, j in zip(image_predictions, message_predictions)]
    return predictions


def create_cnn_model():
    model = Sequential()

    model.add(Convolution2D(64, 3, 3, input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(.2))

    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.2))

    model.add(Convolution2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.2))

    model.add(Flatten())
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error',
                  metrics=['accuracy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                           'cosine_proximity'])
    return model


cnn_model = create_cnn_model()

newModel = cnn_model

from keras.models import load_model

for trainX_image, trainX_message, trainY, testX_image, testX_message, testY in get_data():
    trainY = list(map(lambda x: x * 100, trainY))
    testY = list(map(lambda x: x * 100, testY))

    newModel.fit(np.array(trainX_image),
                 np.array(trainY),
                 validation_data=(
                     np.array(testX_image), np.array(testY)),
                 callbacks=[Static.plot_losses], batch_size=Static.batch_size, epochs=Static.epochs)
    newModel.save("Image_Sentiment_{0}.h5".format(Static.metric_getter.__name__))
