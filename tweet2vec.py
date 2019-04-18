from gensim.models import doc2vec
from gensim import utils
import re

REMOVE_REGEX = [
    '#\w*[a-zA-Z]+\w*',
    '@\w*',
    'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+']

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
        return self.model.infer_vector(tokens)