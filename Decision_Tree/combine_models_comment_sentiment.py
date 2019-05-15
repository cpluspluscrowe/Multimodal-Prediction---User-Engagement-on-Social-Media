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
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Static:
    facebookDb = FacebookDataDatabase()
    metric_getter = facebookDb.getSentiment  # Set this to change the model type
    group_size = 40000
    limit = 60000


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
    return list(map(lambda x: image.load_img(x, target_size=(64, 64)), files))


def transform_image_data(all_files):
    files = list(filter(lambda x: ".jpg" in x or ".png" in x, filter(lambda x: ".DS_Store" not in x, all_files)))
    image_paths = list(map(lambda x: os.path.join("../Image_CNN/images", x), files))
    image_paths = list(filter(lambda x: get_size(x) > 1, image_paths))
    image_arrays = to_array(load_images(image_paths))
    return image_arrays


def get_data():
    all_files = os.listdir("../Image_CNN/images")[:Static.limit]
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
    y_data = list(map(lambda x: Static.metric_getter(x), ids))
    combined_data = zip(image_data, message_data, y_data)
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


for trainX_image, trainX_message, trainY, testX_image, testX_message, testY in get_data():
    image_model = get_model_by_name(Static.metric_getter.__name__)
    message_model = get_model_by_name("nlp_" + Static.metric_getter.__name__)
    train_predictions = get_predictions(trainX_image, trainX_message, trainY, image_model, message_model)
    test_predictions = get_predictions(testX_image, testX_message, testY, image_model, message_model)
    clf_gini = None  # get_model()
    if not clf_gini:
        clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100,
                                          max_depth=5)
    clf_gini.fit(train_predictions, trainY)
    predY = clf_gini.predict(test_predictions)
    print("Accuracy is ", accuracy_score(testY, predY) * 100)
    save_model(clf_gini)
