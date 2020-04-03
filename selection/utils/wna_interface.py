import os
import logging
import pprint
import xml.etree.ElementTree as ET

from nltk.corpus import wordnet as wn

from cfgparser import global_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class WNA_Interface:

    def __init__(self, wna_path):

        self.dict = {}

        tree = ET.parse(wna_path)
        root = tree.getroot()

        for el in root.iter():
            id = el.get('id')
            if id is not None:
                pos, offset = id.split('#')
                self.dict.setdefault(pos, []).append(int(offset))

    def sns_is_emotional(self, synset):
        """ Returns true if synset passed as argument is contained in WNA """
        pos = synset.pos()
        offset = synset.offset()

        try:
            if offset in self.dict[pos]:
                return True
        except KeyError:
            logger.warn('POS {} is not contained in WNA.'.format(pos))
        return False

