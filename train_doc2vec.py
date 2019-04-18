from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import json
import os

IN_PATH = 'tweets_tokenized'
MODEL_PATH = 'models'

if __name__ == '__main__':
    with open('sources.json') as sources_file:
        sources = json.load(sources_file)['sources']
    documents = []
    for source in sources:
        for handle in source['twitter_handles']:
            in_filename = os.path.join(IN_PATH, handle + '.json')
            if not os.path.isfile(in_filename):
                print('No Twitter feed for @' + handle)
                continue
            with open(in_filename) as in_file:
                tweets_json = json.load(in_file)
            print('Processing tweets in file ' + in_filename + '...')

            for tweet in tweets_json['tweets']:
                tokens = tweet['text_tokenized']
                document = TaggedDocument(tokens, [str(tweet['id'])])
                documents.append(document)

    print('Training Doc2Vec model...')
    model = Doc2Vec(documents, vector_size=20, epochs=10, min_count=2)
    model_filename = os.path.join(MODEL_PATH, 'doc2vec')
    model.delete_temporary_training_data(keep_doctags_vectors=False)
    model.save(model_filename)