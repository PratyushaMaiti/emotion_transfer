import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow_hub as hub
import tensorflow as tf
from scipy import spatial

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class SimilarityCalculator:

    def __init__(self, module_url, sess):
        self.embed = hub.Module(module_url)
        self.sess = sess
        self.sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])

    @classmethod
    def return_cosine_similarity(self, vec1, vec2):
        return 1 - spatial.distance.cosine(vec1, vec2)

    def return_embeddings(self, inputs):
        return self.sess.run(self.embed(inputs))

    def return_similarity_scores(self, target_sentence, comparison_sentences):
        if not isinstance(target_sentence, list):
            target_sentence = [target_sentence]

        target_embedding = self.return_embeddings(target_sentence)[0]
        variations_embeddings = self.return_embeddings(comparison_sentences)

        scores = []

        # get similarity scores
        for v_e in variations_embeddings:
            sim = self.return_cosine_similarity(v_e, target_embedding)
            scores.append(sim)

        return scores


def get_similarity_scores(target_sentence, comparison_sentences):
    """ This small helper function is intended to be used when scores should be get in the pipeline """
    # open a Session in a context
    with tf.compat.v1.Session() as session:
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
        # create a SimilarityCalculator instance
        calc = SimilarityCalculator(module_url, session)

        scores = calc.return_similarity_scores(target_sentence, comparison_sentences)

        return [{'sim': score} for score in scores]


if __name__ == '__main__':
    print(get_similarity_scores('What is your age?', ['How old are you?', 'When were you born?', 'I like trains']))