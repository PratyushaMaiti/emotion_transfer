import math
import random
from collections.abc import Sequence
from itertools import chain, combinations

import datasets

import spacy


class Selection:
    """ This class holds information about what tokens are to be substituted in 
    a sequence of tokens """

    def __init__(self, slices):
        """
        Constructor of the Selection class
        
        Keyword arguments:
        - selection:    a selection is a sequence of non-overlapping python slice-objects

        REMINDER:   slice objects in Python have a start and stop attribute. The stop attribute points to the first index
                    which is NOT included in the slice!
        """

        # if argument is single slice, wrap it in a list
        if type(slices) is slice:
            slices = [slices]

        # perform a sanity check of the argument
        # type of selection should be a sequence
        assert type(slices) is tuple or type(slices) is list or type(slices) is slice, "selection argument " \
            + "is not a tuple, list or single slice"
        # all slices should be tuples and have two elements
        assert all(type(sl) is slice for sl in slices), \
            "at least one element in selection which is not a slice"
        # in slices, the second element must be larger than the first (no 'zero' slices)
        assert all(sl.start < sl.stop for sl in slices), "invalid slice found: second element must be" \
            + "larger than the first element"
        # slices should not have step arguments
        assert all(sl.step is None for sl in slices) 
        # slices should not overlap: sort, then check for overlaps
        self.slices = sorted(slices, key=lambda slice: slice.start)

        for i in range(len(self.slices)):
            if i > 0 and self.slices[i].start < self.slices[i-1].stop:
                raise ValueError('Slice {} and {} overlap'.format(
                    self.slices[i],
                    self.slices[i-1]
                ))


    @classmethod
    def from_binary_representation(clf, binary_sequence):
        assert type(binary_sequence) is list or type(binary_sequence) is tuple
        assert all(element in [0,1] for element in binary_sequence)

        slices = []

        for i in range(len(binary_sequence)):
            if binary_sequence[i] == 1:
                slices.append(slice(i, i+1))

        selection = clf(slices)

        return selection


    def __repr__(self):
        # do not print step information -> not needed
        slice_representations = ["slice({},{})".format(sl.start, sl.stop) for sl in self.slices]
        return "<" + ",".join(slice_representations) + ">"

    def return_slices_as_sequences(self, token_sequence, flatten=False):
        """ Takes a sequence of tokens and shows the corresponding token-slices (words, phrases) 
        which are to be substituted given the current selection object
        
        Keyowrd arguments:
        - token_sequence: a list of tokens (strings)
        
        returns:  a list of token sequences, corresponding to the slices defined by selection
        """

        max_slice = max([sl.stop for sl in self.slices])
        seq_length = len(token_sequence)

        assert max_slice <= seq_length, "Selection contains slice " \
            + " which exceeds token sequence length!"
        string_slices = []

        for sl in self.slices:
            if flatten is True and sl.start + 1 == sl.stop:
                string_slices.append(token_sequence[sl][0])
            else:
                string_slices.append(token_sequence[sl])

        return string_slices

    def is_single_words_only(self):
        """ returns true of the selection consists only of slices capturing one single word/token """
        if all(slice.start + 1 == slice.stop for slice in self.slices):
            return True
        return False


class Selector:

    """
    Base class for Selectors. A Selector outputs Selection objects based on the input sequence and other 
    parameters, depending on the used technique
    """

    def __init__(self, input_sequence_tokenized):
        """
        Keyword arguments:

        input_sequence -- a sequence of tokens
        """
        self.input_tokenized = input_sequence_tokenized

        if not isinstance(self.input_tokenized, list) or not all(isinstance(t, str) for t in self.input_tokenized):
            raise ValueError('Input must be a list of strings')

    @classmethod
    def from_spacy_doc(clf, doc):
        assert isinstance(doc, spacy.tokens.doc.Doc)
        sel = clf([token.text for token in doc])
        sel.doc = doc
        return sel

    @classmethod
    def get_powerset_from_slices(clf, slices, max_elements_per_slice):
        """ This method takes a list of slices and returns its powerset (all possible 
        combinations) """

        # powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
        s = list(slices)
        sets = chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))
        return [s for s in sets if len(s) <= max_elements_per_slice]