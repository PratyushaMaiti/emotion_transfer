
"""
Text the emotion classifier's performance with variable input lenghts / batching

"""

import logging

import dill
import time
import torch
import tqdm

import tests.test_config as config
from objective.emotion import EmotionClassifier

logging.basicConfig(filename='performance.log', level=logging.INFO)

model = torch.load(config.emocl_model_path)
model = model.eval()

with open(config.fields_path, 'rb') as f:
    fields = dill.load(f)

ec = EmotionClassifier(model, fields['TEXT'], fields['LABEL'])

input_sentence = 'This is a sentence to test the performance of the classifier'.split()

for i in tqdm.tqdm(range(1, 1000)):
    input_sentences = [input_sentence * i]

    single_batch = ec.return_input_batch(input_sentences)
    time_start = time.time()
    ec.get_outputs(single_batch)
    time_total = time.time()
    logging.info('[{:05d}] single batch:\t{:.2f}'.format(i, time_total - time_start))

    split_batch = ec.yield_input_batches(input_sentences, tokenized=True, batch_size=32)
    time_start = time.time()
    for batch in split_batch:
        ec.get_outputs(batch)
    time_total = time.time()
    logging.info('[{:05d}] multi batch:\t{:.2f}'.format(i, time_total - time_start))
