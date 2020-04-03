from itertools import islice
from pprint import pprint

class NRC_Interface:

    def __init__(self, path):
        self.path = path

    def parse(self):
        self.wordlists = {}
        self.lex = {}
        with open(self.path) as f:
            # first line is empty
            f.readline()
            for line in f:
                word, attribute, indicator = line.split()
                if bool(int(indicator)):
                    if attribute not in self.wordlists.keys():
                        self.wordlists[attribute] = []
                    else:
                        self.wordlists[attribute].append(word)

                self.lex.setdefault(word, {})
                self.lex[word].setdefault(attribute, {})
                self.lex[word][attribute] = bool(int(indicator))

    def is_emotional(self, token):
        if self.lex.get(token):
            if any(self.lex[token].values()):
                return True

        return False


if __name__ == '__main__':
    nrc = NRC_Interface('datasets/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')

    nrc.parse()
    print(nrc.lex['love'])
    print(nrc.is_emotional('love'))

