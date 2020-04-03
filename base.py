from abc import ABC, abstractmethod

import json


class Dataset(ABC):
    """ Base class for handling emotion datasets """

    @abstractmethod
    def make_iterator(self):
        """ This method should return an iterator over all instances in the dataset """
        pass

    @abstractmethod
    def split(self):
        """ This method should split the given dataset and return new Dataset instances according
        to a pre-defined split ratio """
        pass
        


class Selector(ABC):

    @abstractmethod
    def select_tokens(self):
        pass