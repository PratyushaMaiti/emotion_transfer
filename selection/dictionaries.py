import os
import logging

import click
from nltk.corpus import wordnet as wn

from cfgparser import global_config
from preprocessing import SpacyPreprocessor
from selection.selection import Selection, Selector
from selection.utils.wna_interface import WNA_Interface
from selection.utils.nrc_interface import NRC_Interface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class NRC_Selector(Selector):
    def __init__(self, input_sequence_tokenized):

        super().__init__(input_sequence_tokenized)

    @classmethod
    def from_spacy_doc(clf, doc):
        sel = super().from_spacy_doc(doc)
        try:
            nrc_path = os.path.join(global_config['directories']['datasets_dir'], 'NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
            sel.nrc = NRC_Interface(nrc_path)
            # parse NRC
            sel.nrc.parse()
        except FileNotFoundError:
            logger.error('"NRC-Emotion-Lexicon-Wordlevel-v0.92.txt" not found in {}'.format(global_config['directories']['datasets_dir']))
        return sel

    def is_emotional(self):
        selections = self.return_selections()
        if len(selections) == 0:
            return False
        return True

    def return_selections(self):
        # we inherited from the Selector base class, which right now can only be instantiated with
        # a sequence of strings. Therefore, here we have to make sure that the WNA_Selecor was instatiated with
        # the from_spacy_doc classmethod, which gives it the 'wna' interface attribute.
        # Not very elegant, should be changed later
        if not hasattr(self, 'nrc'):
            logger.warning('NRC_Selector instance has no wna interface attribute. It should be instantiated ' \
                'with the "from_spacy_doc" classmethod')
        slices = []
        selected_tokens = []
        for i, token in enumerate(self.doc):
            if self.nrc.is_emotional(token.lemma_):
                # add the corresponding slice to the slices list
                slices.append(slice(i, i+1))
                selected_tokens.append(token)

        logger.info("Tokens {} selected via NRC".format(selected_tokens))

        selections = []
        for set in self.get_powerset_from_slices(slices, 2):
            selections.append(Selection(set))

        return selections

class WNA_Selector(Selector):

    def __init__(self, input_sequence_tokenized):

        super().__init__(input_sequence_tokenized)


        # TODO: this is a duplicate, same attribute is used in the WNRetriever substitution class
        # create a general Spacy -> Wordnet handler to avoid this duplication
        self.pos_mapping = {
            'NOUN': wn.NOUN,
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'VERB': wn.VERB
        }

    @classmethod
    def from_spacy_doc(clf, doc):
        sel = super().from_spacy_doc(doc)
        try:
            wna_path = os.path.join(global_config['directories']['datasets_dir'], 'a-synsets-30.xml')
            sel.wna = WNA_Interface(wna_path)
        except FileNotFoundError:
            logger.error('"a-synsets-30.xml" not found in {}'.format(global_config['directories']['datasets_dir']))
        return sel

    def return_selections(self):
        # we inherited from the Selector base class, which right now can only be instantiated with
        # a sequence of strings. Therefore, here we have to make sure that the WNA_Selecor was instatiated with
        # the from_spacy_doc classmethod, which gives it the 'wna' interface attribute.
        # Not very elegant, should be changed later
        if not hasattr(self, 'wna'):
            logger.warning('WNA_Selector instance has no wna interface attribute. It should be instantiated ' \
                'with the "from_spacy_doc" classmethod')
        slices = []
        for i, token in enumerate(self.doc):
            # first, check if the POS of the token is supported by WN/WNA
            if token.pos_ in self.pos_mapping.keys():
                # get synsets of the current token by lemma and pos
                synsets = wn.synsets(token.lemma_, self.pos_mapping[token.pos_])
                # check, if any of the synsets is contained in WNA ('sns_is_emotional()')
                if any([self.wna.sns_is_emotional(sns) for sns in synsets]):
                    # add the corresponding slice to the slices list
                    slices.append(slice(i, i+1))

        selections = []
        for set in self.get_powerset_from_slices(slices, 2):
            selections.append(Selection(set))

        return selections

    def input_is_emotional(self):
        # checks if a sentence contains at least one token that is contained in WNA
        selections = self.return_selections()
        if len(selections) > 0:
            return True
        return False

def get_percentage_of_emotional_sentences(sentence_list):
    """ This is intended to be a evaluation function to check the coverage of WNA.
    It takes a list of untokenized input sentences and checks for each one if at least one 'emotional'
    token is contained """
    prepr = SpacyPreprocessor()
    counter = 0
    for sentence in sentence_list:
        sentence_doc = prepr.process_input(sentence)
        selector = WNA_Selector.from_spacy_doc(sentence_doc)
        if selector.input_is_emotional():
            counter += 1

    return counter/len(sentence_list) * 100


if __name__ == '__main__':
    path = "inISEAR_sentences.tsv"
    prepr = SpacyPreprocessor()


    with open(path, 'r') as f:
        for i, row in enumerate(f):
            strip = row.strip()
            tokens = prepr.process_input(strip)
            sel = NRC_Selector.from_spacy_doc(tokens)
            if not sel.is_emotional():
                print("Sentence {} is not emotional".format(i))
                print(tokens)

