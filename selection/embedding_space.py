from selection.selection import Selection, Selector
from selection.utils.nrc_interface import NRC_Interface

class EmbeddingSpaceSelector(Selector):

    @classmethod
    def from_spacy_doc(clf, doc, embedding_space, word_lists):
        sel = super().from_spacy_doc(doc)
        sel.embedding_space = embedding_space
        sel.centroids = [sel.embedding_space.get_centroid_from_wordlist(wl) for wl in word_lists]
        return sel

    def word_is_emotional(self, word, threshold):
        vector = self.embedding_space.model.get_vector(word)
        min_distance_to_centroid = self.embedding_space.minimum_distance_to_centroids(vector, self.centroids)
        print('Min distance for {}: {}'.format(word, min_distance_to_centroid))

        if min_distance_to_centroid <= threshold:
            return True

        return False

    def return_selections(self, threshold):
        slices = []
        for i, token in enumerate(self.doc):
            try:
                if self.word_is_emotional(token.lemma_, threshold):
                    slices.append(slice(i, i+1))
            except KeyError:
                continue

        selections = []
        for set in self.get_powerset_from_slices(slices, 2):
            selections.append(Selection(set))

        return selections


if __name__ == "__main__":
    import preprocessing
    import dill
    import torch
    from embedding_space.embedding_space_extractor import EmbeddingSpace

    prepr = preprocessing.SpacyPreprocessor()
    sentence = "I am afraid that I will fail my exam tomorrow"
    sentence_processed = prepr.process_input(sentence)

    model = torch.load('pretrained/tec_no_emb_emocl.pt')
    model.eval()
    embeddings = EmbeddingSpace.return_embedding_weights(model)

    with open('pretrained/tec_no_emb_fields.dill', 'rb') as f:
        fields = dill.load(f)

    emb = EmbeddingSpace.build_from_keyed_vectors(entities=fields['TEXT'].vocab.itos, embedding_weights=embeddings)
    nrc = NRC_Interface('datasets/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')

    sel = EmbeddingSpaceSelector.from_spacy_doc(sentence_processed, emb, nrc.lex.values())
    selections = sel.return_selections(0.25)

    print("Selections:")
    print(selections)

    for selection in selections:
        print(selection.return_slices_as_sequences(sentence_processed))
