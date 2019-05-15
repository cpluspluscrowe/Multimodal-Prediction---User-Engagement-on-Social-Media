import string

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocessing(text):
    text2 = " ".join("".join([" " if ch in string.punctuation else ch for ch in text]).split())
    tokens = [word for sent in nltk.sent_tokenize(text2) for word in nltk.word_tokenize(sent)]
    tokens = [word.lower() for word in tokens]
    stopwds = stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwds]
    tokens = [word for word in tokens if len(word) >= 3]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    tagged_corpus = pos_tag(tokens)
    Noun_tags = ['NN', 'NNP', 'NNPS', 'NNS']
    Verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    lemmatizer = WordNetLemmatizer()

    def prat_lemmatize(token, tag):
        if tag in Noun_tags:
            return lemmatizer.lemmatize(token, 'n')
        elif tag in Verb_tags:
            return lemmatizer.lemmatize(token, 'v')
        else:
            return lemmatizer.lemmatize(token, 'n')

    pre_proc_text = " ".join([prat_lemmatize(token, tag) for token, tag in tagged_corpus])
    return pre_proc_text


class PostText:
    ignore_list = ["imageId", "imageUrl"]

    def __init__(self, post_obj):
        for key in post_obj:
            if not key in self.ignore_list:
                setattr(self, key, post_obj[key])
        self.message = preprocessing(self.message)

    def getValues(self):
        return [x for x in self.__dict__.values()]


def to_vector(column):
    vectorizer = TfidfVectorizer(min_df=4, ngram_range=(1, 3), stop_words='english', max_features=100_000,
                                 strip_accents='ascii', norm='l2', lowercase=True)
    x_data = vectorizer.fit_transform(column).todense()
    return x_data


def test_to_vector():
    import os
    all_files = os.listdir("../Image_CNN/images")
    ids = list(map(lambda x: x[:-4], all_files))
    from Notebooks.LinkDatabases.FacebookData import FacebookDataDatabase
    facebookDb = FacebookDataDatabase()
    metric_getter = facebookDb.getCommentCount  # Set this to change the model type

    rows = list(
        map(lambda x: x[0], filter(lambda x: x if x else None, map(lambda x: facebookDb.getRow(x), ids))))
    data = list(map(lambda x: (x[0], x[10], x[2], x[3]), rows))
    messages = list(map(lambda x: x[1], data))
    word_vectors = to_vector(messages)
    for word_vector in word_vectors[:100]:
        print(word_vector)
