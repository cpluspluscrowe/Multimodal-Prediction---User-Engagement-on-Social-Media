import sys
import os
from itertools import combinations

from keras.applications import VGG16
from scipy.stats import chi2_contingency

sys.path.append("../../")
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, LeakyReLU
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
from keras.callbacks import TensorBoard
from time import time

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Global():
    batch_size = 128
    epochs = 2
    plot_losses = PlotLearning("conv_base_image_cnn_sentiment")


image_count = 50_000  # len(os.listdir("./images"))


def createModel():
    model = Sequential()
    conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(200, 200, 3))
    conv_base.trainable = False
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer=Adam(lr=1e-04), loss='mean_squared_error',
                  metrics=['accuracy', 'mean_squared_error', 'mean_absolute_error'])

    return model


def getSize(filename):
    st = os.stat(filename)
    return st.st_size


def group(iterator, count):
    itr = iter(iterator)
    while True:
        yield tuple([next(itr) for i in range(count)])


def get_ids(files):
    return list(map(lambda x: x.replace(".png", "").replace(".jpg", "").replace("./images/", ""), files))


def to_share_count(id_map, metricGetter):
    return list(map(lambda x: x if x > 0 else 0, map(lambda x: metricGetter(x), id_map)))


def load_images(files):
    final = []
    for file in files:
        try:
            file_loaded = image.load_img(file, target_size=(200, 200))
            final.append(file_loaded)
        except:
            print("Skipping file: {0}".format(file))
            files.remove(file)
    assert len(final) > 1
    return final, files


def to_array(images):
    return np.array(list(map(lambda x: img_to_array(x) / 255, images)))


facebookDb = FacebookDataDatabase()
metricGetter = facebookDb.getSentiment

all_files = os.listdir("./images")[:image_count]
files = list(filter(lambda x: ".jpg" in x or ".png" in x, filter(lambda x: not ".DS_Store" in x, all_files)))
imagePaths = list(map(lambda x: os.path.join("./images", x), files))
imagePaths = list(filter(lambda x: getSize(x) > 1, imagePaths))
imagePaths = list(filter(lambda x: os.path.exists(x), imagePaths))
images, imagePaths = load_images(imagePaths)  # weed out images that fail to load, there are only a few
image_arrays = to_array(images)
shareCounts = to_share_count(get_ids(imagePaths), metricGetter)
labels = np.array(shareCounts)
(trainX, testX, trainY, testY) = train_test_split(image_arrays,
                                                  labels, test_size=0.25, random_state=42)

model_name = '{0}.h5'.format(metricGetter.__name__)
model = createModel()
history = model.fit(trainX, trainY, verbose=0, epochs=Global.epochs, batch_size=Global.batch_size,
                    validation_data=(testX, testY), callbacks=[tensorboard])
print(history.history.keys())
print("loss", history.history["loss"])
print("val_loss", history.history["val_loss"])
print("acc", history.history["acc"])
print("val_acc", history.history["val_acc"])
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


nested_predictions = model.predict(testX)
predictions = list(map(lambda x: x[0], nested_predictions))
truths = testY
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

    if cnt > 100000000:
        exit()
print("Correct: ", correct, " Incorrect: ", incorrect, " Percent Correct: ", correct / (correct + incorrect),
      chi_prob(correct, incorrect))
