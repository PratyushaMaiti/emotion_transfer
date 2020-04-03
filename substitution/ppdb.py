import gzip
import csv
import logging
import re
import time
from collections import namedtuple

from substitution.base import Substitutor

logger = logging.getLogger(__name__)

Paraphrase = namedtuple('Paraphrase', ['pos', 'phrase', 'paraphrase', 'features', 'alignment', 'entailment'])

class PpdbSubstitutor(Substitutor):

    @classmethod
    def from_spacy_doc(clf, doc):
        subst = super().from_spacy_doc(doc)
        subst.cache = dict()
        return subst

    def get_paraphrases(self, token, threshold=10, max_seconds_per_token=30):

        if self.cache.get(token) is None:
            logger.info('Getting paraphrases for token {} from ppdb'.format(token.text.lower()))
            start_time = time.time()

            paraphrases = []
            with gzip.open('datasets/ppdb-2.0-tldr.gz', 'rt') as g:
                for line in g:
                    l = line.strip().split(' ||| ')
                    pos = re.search(r'[A-Z]+', l[0])
                    if pos:
                        pos = pos.group(0)
                    split_at_equal = lambda s: s.split('=')
                    features = {feature:value for feature, value in map(split_at_equal, l[3].split(' '))}
                    phrase, paraphrase, alignment, entailment = l[1], l[2], l[4], l[5]

                    p = Paraphrase(pos, phrase, paraphrase, features, alignment, entailment)

                    if p.phrase == token.text.lower() and p.pos == token.tag_:
                        paraphrases.append(p.paraphrase)
                    if len(paraphrases) >= threshold or time.time() - start_time >= max_seconds_per_token:
                        break
            logger.info('Found paraphrases {}'.format(paraphrases))
            self.cache[token] = paraphrases
            return paraphrases

        else:
            logger.info('Returning paraphrases for "{}" from cache.'.format(token))
            return self.cache.get(token)


    def create_variations(self, selection, conjugate_substitutes):
        return super().create_variations(selection, self.get_paraphrases, conjugate_substitutes)

if __name__ == '__main__':
    from preprocessing import SpacyPreprocessor

    prepr = SpacyPreprocessor()
    word = prepr.process_input('happy')

    subst = PpdbSubstitutor.from_spacy_doc(word)
    subst.get_paraphrases(word[0])

