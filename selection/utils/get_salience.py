"""
Based on Li et. al. (2018), the present script uses the Unified Emotion Dataset (UED) to
get n-grams that are 'salient' for a specific emotion, based on relative counts.
"""

import os
import pprint
from collections import Counter

from tabulate import tabulate
from tqdm import tqdm

from cfgparser import global_config
from objective.emocl.ued_handler import UEDLoader
from preprocessing import SpacyPreprocessor


class SalienceRetriever:

    def __init__(self, **kwargs):
        path_to_ued = os.path.join(global_config['directories']['datasets_dir'], 'unified-dataset.jsonl')
        loader = UEDLoader(path_to_ued)
        self.dataset = loader.filter_datasets(enforce_single=True, **kwargs)
        self.emotion_classes = self.dataset.return_emotion_classes()

    @classmethod
    def return_n_grams(self, n, document, tokenizer):
        tokenized = tokenizer(document)
        n_grams = []

        for i in range(len(tokenized)):
            if i + n > len(tokenized):
                break
            else:
                n_grams.append(tokenized[i:i+n])
        return n_grams

    def get_counts(self, n, tokenizer):
        # for each emotion class, make a counter that keeps counts of the n-grams in documents of the respective emotion class
        counters = {emotion:Counter() for emotion in self.emotion_classes}
        for ex in tqdm(self.dataset.examples):
            emo_class = ex.get_single_emotion()
            n_grams = self.return_n_grams(n, ex.text, tokenizer)
            counters[emo_class].update([" ".join(n_gram).lower() for n_gram in n_grams])

        return counters

    def get_salience(self, counters, smoothing=1, threshold=0, sort=True):
        salient_n_grams = {emotion:list() for emotion in self.emotion_classes}
        for current_emotion, counter in counters.items():
            for n_gram, count in counter.items():
                count_in_other_classes = sum([counters[emotion][n_gram] for emotion in self.emotion_classes if emotion != current_emotion])
                salience = (count + smoothing) / (count_in_other_classes + smoothing)
                if salience > threshold:
                    salient_n_grams[current_emotion].append((n_gram, salience))

        if sort:
            for l in salient_n_grams.values():
                l.sort(key=lambda t: t[1], reverse=True)

        return salient_n_grams

    def test_parameters(self, smoothing_parameters: list, n, k, tokenizer):
        c = self.get_counts(1, tokenizer)
        for p in smoothing_parameters:
            print("Salience scores for smoothing with lambda={}".format(p))
            tabular_data = []
            fieldnames = []
            for emotion in self.emotion_classes:
                fieldnames += ['word ({})'.format(emotion), 'salience ({})'.format(emotion)]
                salient_n_grams = self.get_salience(c, smoothing=p)[emotion]
                if tabular_data == []:
                    tabular_data = salient_n_grams[:k]
                else:
                    for i in range(k):
                        tabular_data[i] += salient_n_grams[i]

            print(tabulate(tabular_data, fieldnames))
            print()



if __name__ == '__main__':
    prepr = SpacyPreprocessor()
    tokenizer = prepr.tokenize

    retr = SalienceRetriever(labeled='single')
#    retr.test_parameters(smoothing_parameters=[0.5, 1.0, 1.5], n=1, k=500, tokenizer=lambda s: s.split())
    c = retr.get_counts(n=1, tokenizer=tokenizer)
    s = retr.get_salience(c, smoothing=0.5, threshold=3)
    pprint.pprint(s)

