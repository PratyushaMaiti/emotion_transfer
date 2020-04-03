import itertools

import spacy

from selection.selection import Selection, Selector


class BruteForceSelector(Selector):
    """ A selector that selects all words for substitution """

    def __init__(self, input_sequence_tokenized):
        super(BruteForceSelector, self).__init__(input_sequence_tokenized)

    def select_one_token(self):
        """ outputs a list of selection objects, with one word marked for substitution each. """
        selections = []
        for i in range(len(self.input_tokenized)):
            selection = Selection(slice(i,i+1)) # select single token in each iteration
            selections.append(selection)

        return selections

    def select_all_combinations_with_threshold(self, threshold=2):
        slices = [selection.slices[0] for selection in self.select_one_token()]
        selections = []
        for set in self.get_powerset_from_slices(slices, max_elements_per_slice=threshold):
            selections.append(Selection(set))

        return selections

    def select_all_combinations(self):
        return self.get_powerset_from_slices(self.select_one_token(), max_elements_per_slice=len(self.input_tokenized))