import sys
import os

sys.path.append("../../")
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
from Notebooks.Scale.MinMaxScalar import ApplyMinMaxScalar
from Notebooks.Plot.plot_training import PlotLearning

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Global():
    batch_size = 1
    epochs = 3
    plot_losses = PlotLearning("image_cnn")


def createModel():
    model = Sequential()

    model.add(Convolution2D(128, 1, 1, input_shape=(128, 128, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.2))

    model.add(Convolution2D(64, 1, 1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.2))

    model.add(Convolution2D(32, 1, 1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.2))

    model.add(Convolution2D(16, 1, 1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(.2))

    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dropout(.2))

    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error',
                  metrics=['accuracy', 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
                           'cosine_proximity'])
    return model


def getSize(filename):
    st = os.stat(filename)
    return st.st_size


def group(iterator, count):
    itr = iter(iterator)
    while True:
        yield tuple([next(itr) for i in range(count)])


image_count = 250000  # len(os.listdir("./images"))
group_size = 240000


def get_ids(files):
    return list(map(lambda x: x.replace(".png", "").replace(".jpg", "").replace("./images/", ""), files))


def to_share_count(id_map, metricGetter):
    return list(map(lambda x: x if x > 0 else 0, map(lambda x: metricGetter(x), id_map)))


def load_images(files):
    return list(map(lambda x: image.load_img(x, target_size=(128, 128), grayscale=True), files))


def to_array(images):
    return np.array(list(map(lambda x: img_to_array(x) / 255, images)))


def data_generator(metricGetter):
    all_files = os.listdir("./images")[:image_count]
    files = list(filter(lambda x: ".jpg" in x or ".png" in x, filter(lambda x: not ".DS_Store" in x, all_files)))
    imagePaths = list(map(lambda x: os.path.join("./images", x), files))
    imagePaths = list(filter(lambda x: getSize(x) > 1, imagePaths))
    for files in group(imagePaths, group_size):
        image_arrays = to_array(load_images(files))
        shareCounts = to_share_count(get_ids(files), metricGetter)
        labels = np.array(shareCounts)
        (trainX, testX, trainY, testY) = train_test_split(image_arrays,
                                                          labels, test_size=0.1, random_state=42)

        yield trainX, trainY, testX, testY


def trainModel(metricGetter):
    model_name = '{0}_simple.h5'.format(metricGetter.__name__)
    model = createModel()

    for trainX, trainY, testX, testY in data_generator(metricGetter):
        print(trainX)
        print(trainY)
        print(testX)
        print(testY)
        print(trainX.shape, trainY.shape)
        model.fit(trainX, trainY, verbose=1, epochs=Global.epochs, batch_size=Global.batch_size,
                  validation_data=(testX, testY), callbacks=[Global.plot_losses])
        model.save(model_name)


facebookDb = FacebookDataDatabase()
trainModel(facebookDb.getCommentCount)

print("Done")
