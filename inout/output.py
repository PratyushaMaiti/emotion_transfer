import csv
import logging

import tabulate

logger = logging.getLogger(__name__)

class OutputGenerator:

    @classmethod
    def to_screen(self, fieldnames, tabular_data):

        print(tabulate.tabulate(tabular_data, fieldnames))

    @classmethod
    def to_csv(clf, fieldnames, tabular_data, outfile):
        writer = csv.writer(outfile, delimiter='\t')
        writer.writerow(fieldnames)
        for row in tabular_data:
            writer.writerow(row)
        logger.info('{} rows written to {}.'.format(len(tabular_data), outfile))

    @classmethod
    def to_tabular_format(clf, original_sentence, sentence_id, variation_sentences, scores, target_scores, target_emotion, config_name, k):
        if not isinstance(target_scores, list) or not isinstance(target_scores, tuple):
#            target_scores = [target_scores]
            pass
        fieldnames = ['ID',  'Sentence Variation'] + [ts for ts in target_scores] + ['t_emotion', 'Config']
        rows = []
        for i, (sentence, score) in enumerate(zip(variation_sentences, scores)):
            row = [str(sentence_id), sentence] + [score[ts] for ts in target_scores] +  [target_emotion, config_name]
            rows.append(row)

            if i == k-1:
                break
        return fieldnames, rows