from substitution.base import Substitutor
from nltk.corpus import wordnet as wn
from preprocessing import SpacyPreprocessor
from selection.selection import Selection
from selection.bruteforce import BruteForceSelector

import itertools
import time

import spacy.tokens

class POSNotSupported(Exception):

    def __init__(self, msg, pos):
        super(POSNotSupported, self).__init__(msg)
        self.args = (pos,)


class WNSubstitutor(Substitutor):

#    def __init__(self, selection):
#        """ Constructor of the class:
#        
#        Keyword arguments:
#        - selection: an instance of the Selection class, indicating which tokens should be substituted
#        """
#
#        if isinstance(selection, Selection):
#            self.selection = selection
#        else:
#            raise TypeError('Argument is not a Selection object!')
#
#        if not self.selection.is_single_words_only():
#            raise NotImplementedError('WordNetSubstitutor only supports substitution of singel words right now.')


    def selection_supports_wn(self, selection):
        pos_mapping = {
            'NOUN': wn.NOUN,
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'VERB': wn.VERB
        }

        tokens_in_selection = selection.return_slices_as_sequences(self.doc, flatten=True)
        pos_supported_by_wn = ['NOUN', 'ADJ','ADV','VERB']
        if not all(token.pos_ in pos_supported_by_wn for token in tokens_in_selection):
            return False
        for token in tokens_in_selection:
            if wn.synsets(token.lemma_, pos_mapping[token.pos_]) == []:
                return False
            # the edgecase to rule all edgecases: there was a case where an ADVERB returned a synset, but no related forms
            token_related = self.retrieve_related_forms(token, 1, 1)
            if len(token_related) == 1 and token_related[0] == token.lemma_:
                return False

        return True

    @classmethod
    def retrieve_related_forms(self, token, move_up, move_down):
        """
        Retrieves a list of related forms for a spacy token objects

        Keyword arguments:
        - token_slice: a list of spacy tokens
        """
        assert type(token) is spacy.tokens.Token

        retr = WNRetriever(token.lemma_, token.pos_)
        return retr.return_all_related_forms(move_up, move_down)

    def create_variations(self, selection, move_up, move_down, conjugate_substitutes):
        related_forms_getter = lambda token: self.retrieve_related_forms(token, move_up, move_down)
        return super().create_variations(selection, related_forms_getter, conjugate_substitutes)


class WNRetriever:
    """ Class that collects related forms from WN for a specific lemma + pos """

    def __init__(self, lemma, pos, synonyms=None, antonyms=None, hyponyms=None, hypernyms=None, similar_words=None):
        # maps POS to respective WN represantation, at the same time indicates available POS
        self.pos_mapping = {
            'NOUN': wn.NOUN,
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'VERB': wn.VERB
        }

        self.lemma = lemma

        # assert that argument POS is supported
        if pos not in self.pos_mapping.keys():
            raise POSNotSupported('The POS "{}" is not supported by WordNet!'.format(pos), pos)
        self.pos = pos
        self.synonyms = synonyms
        self.hyponyms = hyponyms
        self.hypernyms = hypernyms
        self.antonyms = antonyms
        self.similar_words = similar_words

    @classmethod
    def move_up(self, synset, height=0):
        """
        From a synset, move up in WN (relation hypernymy) up to a level indicated by the 'height' argument.
        Aggregate passing synsets along the way.

        Arguments:
        - synset: the synset from whi
        """
        if synset.pos() not in ['v', 'n']:
            raise ValueError('Moving up and hypernomy relations are only supported by verbs and nouns.')

        # put the base synset into a list
        synsets = [synset]

        # initialize variables
        aggregated_synsets = []
        hypernyms = []
        level = 0

        while(level <= height):
            # iterate over synsets in synset list
            for sns in synsets:
                # get hypernyms of synsets in current level
                hypernyms += sns.hypernyms()
                # add them to the aggregation list
                aggregated_synsets += hypernyms
            # if iteration over current level is completed:
            # set synsets list to current hypernyms (will be iterated in next step)
            synsets = hypernyms
            # reset hypernyms
            hypernyms = []
            # increase level
            level += 1

        # remove duplicates by turning into set
        return set(aggregated_synsets)

    @classmethod
    def move_down(self, synset, depth=0):
        """
        From a synset, move down in WN (relation hyponomy) up to a level indicated by the 'depth' argument.
        Aggregate passing synsets along the way.

        Arguments:
        - synset: the synset from whi
        """

        if synset.pos() not in ['v', 'n']:
            raise ValueError('Moving down and hyponomy relations are only supported by verbs and nouns.')

        depth = abs(depth)

        synsets = [synset]

        aggregated_synsets = []
        hyponyms = []
        level = 0

        while(level <= depth):
            for sns in synsets:
                hyponyms += sns.hyponyms()
                aggregated_synsets += hyponyms
            synsets = hyponyms
            hyponyms = []
            level += 1

        return set(aggregated_synsets)

    @classmethod
    def get_lemmas_from_synsets(clf, synsets) -> set:
        """
        Takes a list of synsets and returns all lemma forms that are associated with them

        Arguments:
        - synsets: list of synsets
        """
        lemmas = []
        for sns in synsets:
            lemmas += sns.lemma_names()

        return set(lemmas)

    # TODO: split this method in sub-methods, allow for selection of different aggregation operations 
    # by parameters/config file
    def return_all_related_forms(self, move_up, move_down, similar_antonyms=True, lowercase=True):
        """
        For the lemma/pos pair indicated by the class attributes, return related words 
        by a series of operations.

        Arguments:
        -range: the parameter that is used for the move_up / move_down methods in the taxonomy for verbs/nouns
        -similar_antonyms: whether similar_tos should be applied to antonyms

        """
        related_words = []
        # get all synsets that correspond to lemma/pos combo in attributes
        base_synsets = wn.synsets(self.lemma, self.pos_mapping[self.pos])

        # get all synonyms of these base synsets
        related_words += self.get_lemmas_from_synsets(base_synsets)

        # the following operations depend on the pos
        # different relations in WN are only supported by certain pos

        # moving up, down
        if self.pos == 'NOUN' or self.pos == 'VERB':
            for sns in base_synsets:
                hypernym_synsets = self.move_up(sns, move_up)
                # move_up/down return sequence of synsets, has to be converted to lemma forms first
                related_words += self.get_lemmas_from_synsets(hypernym_synsets)
                hyponym_synsets = self.move_down(sns, move_down)
                related_words += self.get_lemmas_from_synsets(hyponym_synsets)
        elif self.pos == 'ADJ':
            similar_synsets = []
            for sns in base_synsets:
                similar_synsets += sns.similar_tos()
            related_words += self.get_lemmas_from_synsets(set(similar_synsets))

        if self.pos == 'ADJ' or self.pos == 'VERB':
            # antonyms are definded over lemmas objects (we cannot use the string representations aggregated so far here)
            lemmas = []
            for sns in base_synsets:
                lemmas += sns.lemmas()
            antonym_lemmas = []
            for lemma in lemmas:
                antonym_lemmas += lemma.antonyms()
            # add string representations of lemmas to related words list
            related_words += [lemma.name() for lemma in antonym_lemmas]

            if similar_antonyms:
                similar_antonym_synsets = []
                for lemma in antonym_lemmas:
                    similar_antonym_synsets += lemma.synset().similar_tos()
                related_words += self.get_lemmas_from_synsets(set(similar_antonym_synsets))

        if lowercase:
            # make sure forms are lowercase
            lowercased = [word.lower() for word in related_words]
            return lowercased
        
        return related_words

if __name__ == "__main__":

    retriever = WNRetriever('beautiful', 'ADJ')
    print(retriever.return_all_related_forms())