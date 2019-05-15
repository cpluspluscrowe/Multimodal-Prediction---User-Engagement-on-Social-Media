import os
import sys
from sklearn.model_selection import train_test_split

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

sys.path.append(os.pardir)
from keras.models import load_model
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
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten
from keras.models import load_model
from keras.layers import *
from keras.models import Model
import keras
from Notebooks.Plot.plot_training import PlotLearning
from keras.layers import LeakyReLU
from keras import initializers

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Static:
    facebookDb = FacebookDataDatabase()
    metric_getter = facebookDb.getCommentCount  # Set this to change the model type
    limit = 80_000  # len(os.listdir("../Image_CNN/images"))
    group_size = limit
    plot_losses = PlotLearning("combined_keras_model_comment_count")
    batch_size = 512
    epochs = 100


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
            load = image.load_img(file_to_load, target_size=(28, 28), grayscale=True)
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
    all_files = os.listdir("../Image_CNN/images")[:Static.limit]
    ids = list(map(lambda x: x[:-4], all_files))
    rows = list(
        map(lambda x: x[0], filter(lambda x: x if x else None, map(lambda x: Static.facebookDb.getRow(x), ids))))
    data = list(map(lambda x: (x[0], x[10], x[2], x[3]), rows))
    messages = list(map(lambda x: x[1], data))
    image_data = transform_image_data(all_files)
    message_data = to_vector(messages)
    y_data_metrics = list(map(lambda x: x if x > 0 else 0,
                              filter(lambda x: x != None, map(lambda x: Static.metric_getter(x), ids))))
    y_data = list(map(lambda x: x, y_data_metrics))
    combined_data = zip(image_data, message_data, y_data)
    if Static.group_size is None:
        Static.group_size = len(list(combined_data))
    import statistics
    print("Data Var: ", statistics.stdev(y_data) ** 2)
    print(y_data)
    image_data_batch, message_data_batch, y_data_batch = zip(*combined_data)
    (trainX_image, testX_image, trainY_image, testY_image) = train_test_split(image_data_batch, y_data_batch,
                                                                              test_size=0.1, random_state=42)
    (trainX_message, testX_message, trainY_message, testY_message) = train_test_split(message_data_batch,
                                                                                      y_data_batch, test_size=0.1,
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def get_predictions(image_data, message_data, answers, image_model, message_model):
    message_data_squeezed = np.squeeze(np.array(message_data), axis=1)
    image_predictions = image_model.predict(np.array(image_data)).tolist()
    message_predictions = message_model.predict(message_data_squeezed).tolist()
    predictions = [[i[0], j[0]] for i, j in zip(image_predictions, message_predictions)]
    return predictions


def create_nlp_model():
    model = Sequential()
    input_dim = 64197
    model.add(Dense(1024, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    model.add(Dense(10, input_dim=input_dim))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.2))

    return model


def create_cnn_model():
    model = Sequential()

    model.add(Convolution2D(1024, 2, 1, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.1))

    model.add(Convolution2D(32, 2, 1))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.1))

    model.add(Flatten())
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(.1))

    return model


nlp_model = create_nlp_model()
cnn_model = create_cnn_model()

mergedOut = keras.layers.Add()([cnn_model.output, nlp_model.output])
mergedOut = Dense(10)(mergedOut)
mergedOut = LeakyReLU(alpha=0.1)(mergedOut)
mergedOut = Dropout(.1)(mergedOut)

mergedOut = Dense(1, activation='sigmoid')(mergedOut)
newModel = Model([cnn_model.input, nlp_model.input], outputs=mergedOut)
newModel.compile(optimizer=Adam(), loss='mean_squared_error',
                 metrics=['accuracy'])

model_path = "Combined_Model_{0}.h5".format(Static.metric_getter.__name__)

trainX_image, trainX_message, trainY, testX_image, testX_message, testY = get_data()
history = newModel.fit([np.array(trainX_image), np.squeeze(np.array(trainX_message), axis=1)],
                       np.array(trainY),
                       validation_data=(
                           [np.array(testX_image), np.squeeze(np.array(testX_message), axis=1)], np.array(testY)),
                       callbacks=[Static.plot_losses], batch_size=Static.batch_size, epochs=Static.epochs)

newModel.save("Combined_Model_{0}.h5".format(Static.metric_getter.__name__))
