import os
from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
import re
import matplotlib.pyplot as plt
import pandas as pd


def visualize_word2vec_model(path):
    model_name = os.path.basename(path)
    model = Word2Vec.load(path)

    vocab = list(model.wv.vocab)
    X = model[vocab]

    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)

    df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Word2Vec Visualization")

    ax.scatter(df['x'], df['y'])
    for word, pos in df.iterrows():
        ax.annotate(word, pos)

    fig.savefig("./{0}".format(model_name.replace("h5", "png")))


visualize_word2vec_model(r"/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Word2Vec/highly_positive.h5")
visualize_word2vec_model(r"/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Word2Vec/slightly_positive.h5")
visualize_word2vec_model(r"/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Word2Vec/negative_positive.h5")
