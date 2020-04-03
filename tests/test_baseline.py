import random

import dill
import pytest
import torch

from tests.test_dataset import ued_dataset
from objective.emocl.ued_handler import UEDLoader
from baseline import BaseLine


def test_variation_generation_untokenized(ued_dataset):
    # sample random tweets from ued
    sample_inputs = [example.text for example in random.sample(ued_dataset.examples, 10)]
    for sample in sample_inputs:
        BaseLine.get_variations(sample)

def test_baseline(ued_dataset):

    model = torch.load('emoclass.pt')
    with open('fields.dill', 'rb') as f:
        fields = dill.load(f)
    sample_inputs = [example.text for example in random.sample(ued_dataset.examples, 100)]
    for sample in sample_inputs:
        variations = BaseLine.get_variations(sample)
        if variations == []:
            with pytest.raises(ValueError):
                scores = BaseLine.annotate_variatons(variations, model, fields['TEXT'], fields['LABEL'])
        else:
            scores = BaseLine.annotate_variatons(variations, model, fields['TEXT'], fields['LABEL'])