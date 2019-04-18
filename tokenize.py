from gensim import utils
import json
import os
import re

IN_PATH = 'tweets_tidy'
OUT_PATH = 'tweets_tokenized'
REMOVE_REGEX = [
    '#\w*[a-zA-Z]+\w*',
    '@\w*',
    'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+']

if __name__ == '__main__':
    with open('sources.json') as sources_file:
        sources = json.load(sources_file)['sources']
    for source in sources:
        for handle in source['twitter_handles']:
            in_filename = os.path.join(IN_PATH, handle + '.json')
            if not os.path.isfile(in_filename):
                print('No Twitter feed for @' + handle)
                continue
            with open(in_filename) as in_file:
                tweets_json = json.load(in_file)
            print('Tokenizing tweets in file ' + in_filename + '...')

            for tweet in tweets_json['tweets']:
                text_in = tweet['text']
                text_out = text_in
                for regex in REMOVE_REGEX:
                    text_out = re.sub(regex, '', text_out)
                text_tokenized = utils.simple_preprocess(text_out, deacc=True)
                tweet['text_tokenized'] = text_tokenized

            out_filename = os.path.join(OUT_PATH, handle + '.json')
            with open(out_filename, 'w') as out_file:
                json.dump(tweets_json, out_file, indent=2)