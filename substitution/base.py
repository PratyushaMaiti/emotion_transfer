from nltk.corpus import wordnet as wn
from preprocessing import SpacyPreprocessor
from selection.selection import Selection
from selection.bruteforce import BruteForceSelector

import itertools
import logging
import time

import spacy.tokens

logger = logging.getLogger(__name__)

try:
    from pattern.en import conjugate
except ImportError:
    logger.warning('Could not import pattern, conjugation will not work.')

class POSNotSupported(Exception):

    def __init__(self, msg, pos):
        super(POSNotSupported, self).__init__(msg)
        self.args = (pos,)


class Substitutor:

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


    @classmethod
    def from_spacy_doc(clf, doc):
        """ This returns a class instance from a spacy DOC instance
        
        Keyword arguments:
        - doc: a spacy doc object
        - selection: an instance of the Selection class, indicating which tokens should be substituted
        """

        assert type(doc) is spacy.tokens.doc.Doc, "Argument is not a valid spacy doc instance!"
        subst = clf()
        subst.doc = doc
        return subst


    def create_variations(self, selection, related_forms_getter, conjugate_substitutes=False):
        # check if selection conforms to the token sequence boundaries
        if max([slice.stop for slice in selection.slices]) > len(self.doc):
            raise IndexError("selection contains slice going beyond token boundaries.")

        # this list holds all variations that are created through the substitutions produced by the current selection
        variations = []

        target_tokens = selection.return_slices_as_sequences(self.doc, flatten=True)

        # for each token slice, generate a set of related forms, put them in a nested list
        related_forms_lists = [set(related_forms_getter(token)) for token in target_tokens]
        related_forms_lists_filtered = []

        # assert that the related forms lists do not contain the original token
        for t, rel in zip(target_tokens, related_forms_lists):
            filtered = list(filter(lambda a: a != t.text, rel))
            related_forms_lists_filtered.append(filtered)

        # itertools.product outputs the cartesian product of all the arguments
        # in this case: all prossible combinations of substitutions!
        # the 'substitute list' in each iteration contains one individual set of combination of 
        for substitute_list in itertools.product(*related_forms_lists_filtered):
            # make a copy of the original input token sequence
            token_sequence_str = [token.lower_ for token in self.doc]
            insertion_points = [slice.start for slice in selection.slices]
            for i, (insertion_point, target_token, substitute) in enumerate(zip(insertion_points, target_tokens, substitute_list)):
                assert substitute is not None

                # delete target token at insertion point
                del(token_sequence_str[insertion_point])

                # split the current substitute in case it is a compound
                phrase  = substitute.split('_')

                # if current target is NOT a phrasal verb, try to conjugate it accordingly
                if conjugate_substitutes and target_token.pos_ == 'VERB' and len(phrase) == 1:
                    try:
                        phrase = [conjugate(phrase[0], target_token.tag_)]
                    except NameError:
                        logger.warning('Conjugation function not available. Is pattern imported correctly?')
                    except:
                        logger.warning('Conjugating phrase "{}" with tag "{}" failed.'.format(phrase, target_token.tag_))

                # TODO: add plurilazation for nouns

                # insert the new word form(s) at the given index
                for j in range(len(phrase)):
                    token_sequence_str.insert(insertion_point+j, phrase[j])

                # if substitute is a phrase, move the following insertion point to the right
                if i < len(insertion_points)-1 and len(phrase) > 1:
                    insertion_points[i+1] += len(phrase) - 1

            if all(token is not None for token in token_sequence_str):
                variations.append(token_sequence_str)
        
        return variations

