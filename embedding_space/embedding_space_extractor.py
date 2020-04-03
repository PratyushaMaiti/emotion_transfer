import logging
import os

import dill
import numpy as np
import tabulate
import torch
from gensim.models import KeyedVectors
from gensim.similarities.index import AnnoyIndexer
from scipy.spatial.distance import cosine

from selection.utils.nrc_interface import NRC_Interface

logging.basicConfig(level=logging.INFO)

class EmbeddingSpace:

    @classmethod
    def build_from_keyed_vectors(clf, entities, embedding_weights, annoy_trees=10):
        emb = clf()
        emb.model = emb.get_keyed_vectors(entities, embedding_weights)
        
        emb.indexer = AnnoyIndexer(emb.model, annoy_trees)
        emb.indexer.build_from_keyedvectors()
        return emb

    @classmethod
    def build_from_word2vec(clf, path_to_w2v, binary, annoy_trees=10):
        emb = clf()
        logging.info('Loading word2vec from {}'.format(path_to_w2v))
        emb.model = KeyedVectors.load_word2vec_format(path_to_w2v, binary=binary)

        logging.info('Initializing index ...')
        emb.indexer = AnnoyIndexer(emb.model, annoy_trees)
        return emb

    @classmethod
    def load_from_files(clf, model_path, indexer_path):
        emb = clf()
        emb.model = KeyedVectors.load(model_path)
        emb.indexer = AnnoyIndexer()
        emb.indexer.load(indexer_path)
        return emb

    @classmethod
    def return_embedding_weights(
        self, 
        emocl_model, 
        embeddings_key='feature_extractor.embedding.embedding.weight'):
        return emocl_model.state_dict()[embeddings_key].cpu().numpy()

    @classmethod
    def get_keyed_vectors(clf, entities, embedding_weights):
        embeddings_dim = embedding_weights.shape[1]
        word_vectors = KeyedVectors(embeddings_dim)
        word_vectors.add(entities=entities, weights=embedding_weights)
        return word_vectors

    def save_indexer(self, output_path):
        self.indexer.save(output_path)
        logging.info('Index saved to {}'.format(output_path))

    def save_model(self, output_path):
        self.model.save(output_path)
        logging.info('Model saved to {}'.format(output_path))

    def get_centroid_from_wordlist(self, wordlist):
        wordlist_embeddings = []
        for word in wordlist:
            try:
                emb = self.model.get_vector(word)
            except KeyError:
                continue
            else:
                wordlist_embeddings.append(emb)

        # calculate the centroid
        return np.mean(wordlist_embeddings, axis=0)

    def most_similar(self, word, topn):
        return self.model.most_similar(word, topn=topn, indexer=self.indexer)

    def highest_distance_to_centroids(self, vector, centroid_vectors):
        # returns the highest similarity to any of the given centroids
        # the higher the cosine similarity, the less is the distance between two vectors
        if not isinstance(centroid_vectors, np.ndarray):
            centroid_vectors = np.array(centroid_vectors)
        # returns cosine similarities
        return max([1 - sim for sim in self.model.cosine_similarities(vector, centroid_vectors)])


if __name__ == '__main__':
    import tabulate
    model = torch.load('pretrained/emoint_tw_emb_emocl.pt')
    embeddings = EmbeddingSpace.return_embedding_weights(model)

    with open('pretrained/emoint_tw_emb_fields.dill', 'rb') as f:
        fields = dill.load(f)
    emb = EmbeddingSpace.build_from_keyed_vectors(entities=fields['TEXT'].vocab.itos, embedding_weights=embeddings)

    nrc = NRC_Interface('datasets/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    centroids = [emb.get_centroid_from_wordlist(word_list) for word_list in nrc.lex.values()]

    joy_centroid = emb.get_centroid_from_wordlist(nrc.lex['sadness'])
    print(emb.model.similar_by_vector(joy_centroid, topn=50))



    word_list = ['chair', 'mug', 'jacket', 'scary', 'hopeless', 'anxiety', 'racist', 'fear', 'fuck', 'shit', 'depression', 'alone']
    word_vectors = [emb.model.get_vector(word) for word in word_list]
    highest_distances = [emb.highest_distance_to_centroids(vector, centroids) for vector in word_vectors]

    print(tabulate.tabulate(tabular_data=zip(word_list, highest_distances), headers=['word', 'max_similarity']))

