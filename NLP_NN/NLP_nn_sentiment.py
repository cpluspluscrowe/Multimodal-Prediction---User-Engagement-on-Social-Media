import os
import sys

sys.path.append(os.pardir)
from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase

import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import numpy as np
from Notebooks.Scale.MinMaxScalar import ApplyMinMaxScalar
from sklearn.model_selection import train_test_split
from Notebooks.Text.Process import PostText, to_vector
from Notebooks.Mapper.Message_Sentiment import MessageGetter
from Notebooks.Plot.plot_training import PlotLearning
from keras import regularizers


class Global:
    batch_size = 1
    epochs = 5
    group_size = 6000
    plot_losses = PlotLearning("nlp_nn")
    facebookDb = FacebookDataDatabase()
    metric_getter = facebookDb.getSentiment
    metric_name = None
    regularizer_function = None


def identity(val):
    return val


if "Share" in Global.metric_getter.__name__:
    Global.metric_name = "shareCount"
elif "Comment" in Global.metric_getter.__name__:
    Global.metric_name = "commentCount"
elif "Sentiment" in Global.metric_getter.__name__:
    Global.metric_name = "postPositivity"
else:
    raise Exception("Did not get the right metric name")

postTexts = []

for post in MessageGetter.get_post_generator():
    pt = PostText(post)
    count = pt.getValues()
    postTexts.append(pt.getValues())

df = pd.DataFrame.from_records(postTexts, columns=MessageGetter.get_columns())
df = df[df[Global.metric_name] > 0]  # FILTER OUT ZEROS FOR NOW
x_data = to_vector(df["message"])
y_data = list(map(lambda x: x if x > 0 else 0, df[Global.metric_name]))
print(len(x_data))
print(len(y_data))

model = Sequential()


def create_model():
    input_layer_count = 1  # y_data.shape[1]
    model.add(Dense(200, input_dim=x_data.shape[1]))
    model.add(Dropout(0.3))

    model.add(Dense(100, input_dim=x_data.shape[1]))
    model.add(Dropout(0.3))

    model.add(Dense(1))
    model.add(Dropout(0.3))
    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy', 'mean_squared_error'])
    return model


def group(iterator, count):
    itr = iter(iterator)
    while True:
        yield tuple([next(itr) for i in range(count)])


def data_generator(metricGetter):
    counts = np.array(y_data)
    if Global.group_size is None:
        Global.group_size = len(counts)
    for batch in group(list(map(lambda x, y: (x, y), x_data, counts)), Global.group_size):
        x_batch, y_batch = zip(*batch)
        x_batch = np.squeeze(np.array(x_batch), axis=1)
        y_batch = np.array(y_batch)
        (trainX, testX, trainY, testY) = train_test_split(x_batch, y_batch, test_size=0.25, random_state=42)
        yield trainX, trainY, testX, testY


def trainModel(metricGetter):
    model_name = 'nlp_{0}.h5'.format(metricGetter.__name__)
    # if os.path.exists(model_name):
    #    print("Loading model")
    #    model = load_model(model_name)
    # else:
    if True:
        print("Creating model")
        model = create_model()

    for trainX, trainY, testX, testY in data_generator(metricGetter):
        trainY = list(map(lambda x: x * 100, trainY))
        testY = list(map(lambda x: x * 100, testY))
        history = model.fit(trainX, trainY, verbose=1, epochs=Global.epochs, batch_size=Global.batch_size,
                            validation_data=(testX, testY), callbacks=[Global.plot_losses])
        model.save(model_name)


trainModel(Global.metric_getter)
print("Finished")
