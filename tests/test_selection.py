import pytest
from selection import BruteForceSelector, Selection

def test_allowed():
    # this should work:
    seq = [
        slice(0,1), # first element
        slice(1,2), # second element
        slice(2,4), 
        slice(5,7)
    ]
    selection = Selection(seq)

def test_overlap():
    with pytest.raises(ValueError):
        seq = [
            slice(0,1),
            slice(0,3),
            slice(1,3)
        ]
        selection = Selection(seq)

def test_overlap_unordered():
    with pytest.raises(ValueError):
        seq = [
            slice(1,3),
            slice(0,1),
            slice(0,3),
        ]
        selection = Selection(seq)

def test_return_words():
    token_sequence = ['This', 'is', 'a', 'test', '!']
    selection = Selection([
        slice(0,1), # This
        slice(2,4), # a test
    ])
    target_tokens = [['This'], ['a', 'test']]

    assert selection.return_slices_as_sequences(token_sequence) == target_tokens

def test_selection_too_long():
    with pytest.raises(AssertionError):
        token_sequence = ['This', 'is', 'a', 'test', '!']
        selection = Selection([
            slice(0,1), # This
            slice(2,4), # a test
            slice(4,6)  # exceeds length of token sequence -> should throw error
        ])
        selection.return_slices_as_sequences(token_sequence)

def test_bf_selection():
    token_sequence = ['This', 'is', 'a', 'test', '!']
    gold_target_sequences = [
        ['This'],
        ['is'],
        ['a'],
        ['test'],
        ['!']
    ]
    bfs = BruteForceSelector(token_sequence)

    # returns a list of selection objects, each targeting one token
    selections = bfs.select_one_token()
    # let each selection object return its token slice (here: one word), put them in a list
    generated_target_sequences = []
    for slice in selections:
        generated_target_sequences += slice.return_slices_as_sequences(token_sequence)

    assert gold_target_sequences == generated_target_sequences

def test_selection_from_binary():
    binary_sequence = [0,0,1,1,0]
    gold_selection = [
        slice(2,3),
        slice(3,4)
    ]

    selection = Selection.from_binary_representation(binary_sequence)
    assert selection.slices == gold_selection