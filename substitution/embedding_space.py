import dill
import numpy as np
import torch

from preprocessing import SpacyPreprocessor
from embedding_space.embedding_space_extractor import EmbeddingSpace
from selection.utils.nrc_interface import NRC_Interface
from substitution.base import Substitutor


class OutOfVocabulary(Exception):
    
    def __init__(self, token):
        msg = "Token '{}' is not present in the vocabulary of the embedding space.".format(token.text)
        super().__init__(msg)


class EmbeddingSpaceSubstitutor(Substitutor):

    @classmethod
    def from_spacy_doc(clf, doc, embedding_space):

        subst = super().from_spacy_doc(doc)
        subst.embedding_space = embedding_space

        return subst

    def get_neighboring_words(self, token, n_neighbors):
        try:
            similar_words = [word for word, _ in self.embedding_space.most_similar(token.text.lower(), n_neighbors)]
        except KeyError:
            raise OutOfVocabulary(token)

        return similar_words


    def create_variations(self, selection, n_neighbors, conjugate_substitutes):
        related_forms_getter = lambda word: self.get_neighboring_words(word, n_neighbors)
        return super().create_variations(selection, related_forms_getter, conjugate_substitutes)


class InformedEmbeddingSpaceSubstitutor(EmbeddingSpaceSubstitutor, Substitutor):

    @classmethod
    def from_spacy_doc(clf, doc, embedding_space, lexicon):

        subst = super().from_spacy_doc(doc, embedding_space)

        # an emotion lexicon is a dict with emotions as its keys
        # and values are word lists
        subst.lexicon = lexicon

        # calculate centroids according to keys (emotions) in the lexicon
        subst.centroids = {emotion: subst.embedding_space.get_centroid_from_wordlist(subst.lexicon[emotion]) for emotion in subst.lexicon}

        return subst

    def get_words_closest_to_target_emotion(self, token, target_emotion, n_neighbors, n_least_distance_to_centroid, deduct_average_dist=True):
        similar_words = EmbeddingSpaceSubstitutor.get_neighboring_words(self, token, n_neighbors)

        similar_words_vectors = np.array([self.embedding_space.model.get_vector(word) for word in similar_words])
        similarities_to_target_emotion = self.embedding_space.model.cosine_similarities(
            self.centroids[target_emotion],
            similar_words_vectors)

        avg_distances_to_other_emotions = []
        for vec in similar_words_vectors:
            other_centroids = [centroid for emotion, centroid in self.centroids.items() if emotion != target_emotion]
            distances_to_other_centroids = self.embedding_space.model.cosine_similarities(vec, other_centroids)
            avg_distances_to_other_emotions.append(sum(distances_to_other_centroids)/len(other_centroids))

        if deduct_average_dist:
            similar_words_sorted = sorted(
                zip(similar_words, similarities_to_target_emotion, avg_distances_to_other_emotions), 
                key=lambda t: t[1] - t[2], reverse=True)
            return [word for word, _, _ in similar_words_sorted][:n_least_distance_to_centroid]
        else:
            similar_words_sorted = sorted(
                zip(similar_words, similarities_to_target_emotion), 
                key=lambda t: t[1], reverse=True)
            return [word for word, _ in similar_words_sorted][:n_least_distance_to_centroid]

    def create_variations(self, selection, emotion, n_neighbors, n_least_distance_to_centroid, conjugate_substitutes):
        related_forms_getter = lambda word: self.get_words_closest_to_target_emotion(word, emotion, n_neighbors, n_least_distance_to_centroid)
        return Substitutor.create_variations(self, selection, related_forms_getter, conjugate_substitutes)


if __name__ == '__main__':
    model = torch.load('pretrained/tec_emocl.pt')
    model.eval()
    embeddings = EmbeddingSpace.return_embedding_weights(model)

    with open('pretrained/tec_fields.dill', 'rb') as f:
        fields = dill.load(f)
    emb = EmbeddingSpace.build_from_keyed_vectors(entities=fields['TEXT'].vocab.itos, embedding_weights=embeddings)
    nrc = NRC_Interface('datasets/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    nrc.parse()

    prepr = SpacyPreprocessor()
    doc = prepr.process_input("girl")

    subst = EmbeddingSpaceSubstitutor.from_spacy_doc(doc, emb, nrc.wordlists)
    print("With deduction of avg distances:")
    print(subst.get_words_closest_to_target_emotion(doc[0], 'joy', 50, 50))
    print("Without deduction of avg distances:")
    print(subst.get_words_closest_to_target_emotion(doc[0], 'joy', 50, 50, deduct_average_dist=False))
