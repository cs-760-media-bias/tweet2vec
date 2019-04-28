from gensim.models import doc2vec
from gensim.models import word2vec
from gensim import utils
import numpy as np
from numpy.linalg import norm
import re

REMOVE_REGEX = [
    r'#\w*[a-zA-Z]+\w*',
    r'@\w*',
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+']


def tokenize(tweet_text):
    text_out = tweet_text
    for regex in REMOVE_REGEX:
        text_out = re.sub(regex, '', text_out)
    return utils.simple_preprocess(text_out, deacc=True)


class Doc2Vec():
    def __init__(self, model_filename):
        self.model = doc2vec.Doc2Vec.load(model_filename)

    def vectorize(self, tweet_text):
        tokens = tokenize(tweet_text)
        if len(tokens) == 0:
            raise ValueError('zero tokens')
        return self.model.infer_vector(tokens)


class Word2Vec():
    def __init__(self, model_filename):
        self.model = word2vec.Word2Vec.load(model_filename)

    def vectorize(self, tweet_text):
        tokens = tokenize(tweet_text)
        if len(tokens) == 0:
            raise ValueError('zero tokens')
        total = np.zeros(self.model.vector_size)
        for token in tokens:
            total += self.model.wv[token]
        return total / norm(total)
