"""
Get some random sentences from UED to create a sample input file for 
the emotion transfer script
"""

import os
from spacy.pipeline import SentenceSegmenter
from preprocessing import SpacyPreprocessor

from objective.emocl.ued_handler import UEDLoader
from cfgparser import global_config

def get_random_sentences(output, num=1000, **kwargs):
    prepr = SpacyPreprocessor()
    ued_path = os.path.join(global_config['directories']['datasets_dir'], 'unified-dataset.jsonl')
    loader = UEDLoader(ued_path)
    data = loader.filter_datasets(**kwargs)

    examples = []
    while len(examples) < num:
        example = prepr.process_input(data.return_random_example().text)
        if len(list(example.sents)) == 1 and example.text not in examples:
            examples.append(example.text + "\n")

    with open(output, 'w') as f:
        f.writelines(examples)


if __name__ == '__main__':
    get_random_sentences('tec_examples.txt', source='tec')