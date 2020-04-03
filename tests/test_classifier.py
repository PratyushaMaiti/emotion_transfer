import random
import string

import dill
import pytest
import torch

import test_config as config
from test_dataset import ued_dataset
from objective.emotion import EmotionClassifier


@pytest.fixture
def ec_model():
    model = torch.load(config.emocl_model_path)
    return model.eval()

@pytest.fixture
def fields():
    with open(config.fields_path, 'rb') as f:
        return dill.load(f)

def test_get_scores_untokenized(ued_dataset, ec_model, fields):
    random.seed(42)

    sample_inputs = [example.text for example in random.sample(ued_dataset.examples, 100)]
    ec = EmotionClassifier(ec_model, fields['TEXT'], fields['LABEL'])
    ec.get_scores(sample_inputs)

def test_get_scores_from_scrambeled(ec_model, fields):
    scrambled_sequence = []
    for i in range(10):
        scrambled_string = "".join(random.choices(string.ascii_letters + string.punctuation, k=10))
        scrambled_sequence.append(scrambled_string)

    ec = EmotionClassifier(ec_model, fields['TEXT'], fields['LABEL'])
    ec.get_scores(scrambled_sequence)

