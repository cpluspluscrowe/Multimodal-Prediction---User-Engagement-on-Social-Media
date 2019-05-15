from keras.utils import plot_model
import os
from keras.models import load_model

path = "/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Combine_Keras_Models/Combined_Model_getCommentCount.h5"
model_name = os.path.basename(path)
model = load_model(path)

plot_model(model, to_file='{0}.png'.format(model_name))
