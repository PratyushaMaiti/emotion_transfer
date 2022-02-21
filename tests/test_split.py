import pytest


@pytest.fixture
def tec_dataset():
    from objective.emocl.ued_handler import UEDLoader
    loader = UEDLoader('datasets/unified-dataset.jsonl')
    return loader.filter_datasets(source='tec')

def test_empty(tec_dataset):
    with pytest.raises(ValueError):
        train, test = tec_dataset.split(ratio=())

def test_micro_dataset(tec_dataset):
    from objective.emocl.ued_handler import UEDDataset

    # create a micro dataset
    small = UEDDataset(tec_dataset.examples[:1])

    with pytest.raises(ValueError):
        train, test = small.split()

