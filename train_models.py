from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.word2vec import Word2Vec
import json
import os
import tweet2vec

IN_PATH = 'tweets_tidy'
MODEL_PATH = 'models'

if __name__ == '__main__':
    with open('sources.json') as sources_file:
        sources = json.load(sources_file)['sources']
    docs = []
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
                tokens = tweet2vec.tokenize(tweet['text'])
                doc = TaggedDocument(tokens, [str(tweet['id'])])
                docs.append(doc)

    print('Training Doc2Vec model...')
    doc2vec = Doc2Vec(docs,
        epochs=5,       # Number of training epochs
        min_count=5,    # Minimum count of words to consider
        vector_size=20, # Dimensionality of vectors
        workers=10)     # Number of worker threads
    doc2vec_filename = os.path.join(MODEL_PATH, 'doc2vec')
    doc2vec.delete_temporary_training_data(keep_doctags_vectors=False)
    doc2vec.save(doc2vec_filename)

    print('Training Word2Vec skip-gram model...')
    word2vec_sg = Word2Vec([doc.words for doc in docs],
        iter=5,         # Number of training epochs
        min_count=5,    # Minimum count of words to consider
        sg=1,           # 1 for skip-gram, 0 for CBOW
        size=20,        # Dimensionality of vectors
        workers=10)     # Number of worker threads   
    word2vec_sg_filename = os.path.join(MODEL_PATH, 'word2vec_sg')
    word2vec_sg.delete_temporary_training_data()
    word2vec_sg.save(word2vec_sg_filename)

    print('Training Word2Vec CBOW model...')
    word2vec_cbow = Word2Vec([doc.words for doc in docs],
        iter=5,         # Number of training epochs
        min_count=5,    # Minimum count of words to consider
        sg=0,           # 1 for skip-gram, 0 for CBOW
        size=20,        # Dimensionality of vectors
        workers=10)     # Number of worker threads  
    word2vec_cbow_filename = os.path.join(MODEL_PATH, 'word2vec_cbow')
    word2vec_cbow.delete_temporary_training_data()
    word2vec_cbow.save(word2vec_cbow_filename)