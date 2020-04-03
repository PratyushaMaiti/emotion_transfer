import pytest
from objective.emocl.ued_handler import UEDDataset

@pytest.fixture
def ued_dataset():
    from objective.emocl.ued_handler import UEDLoader
    loader = UEDLoader('/home/dave/Development/Python/emotion-transfer/emotion-transfer/datasets/unified-dataset.jsonl')
    ued = loader.filter_datasets()
    return ued

def test_invalid_init():
    """ pass non-example data to dataset constructor """
    list = [1,2,3,4,5]

    with pytest.raises(AssertionError):
        dataset = UEDDataset(list)
    
def test_token_access():
    """ try to access tokens without having tokenized """
    