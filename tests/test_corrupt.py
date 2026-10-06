import random

import pytest

from newline_fixer.corrupt import corrupt, render
from newline_fixer.labels import Label, decode, encode, split_input

TEXT = (
    "3.2.3 Applications of Attention in our Model\n\n"
    "The Transformer uses multi-head attention in three different ways:\n"
    "• In encoder-decoder attention layers, the queries come from the previous decoder layer, "
    "and the memory keys and values come from the output of the encoder.\n\n"
    "This allows every position in the decoder to attend over all positions in the input sequence."
)


@pytest.mark.parametrize("seed", range(50))
def test_gold_labels_restore_original(seed):
    sample = corrupt(TEXT, random.Random(seed))
    assert decode(sample["words"], sample["labels"]) == decode(*encode(TEXT))


@pytest.mark.parametrize("seed", range(50))
def test_join_only_at_line_breaks(seed):
    sample = corrupt(TEXT, random.Random(seed), split_prob=1.0)
    assert len(sample["breaks"]) == len(sample["labels"]) == len(sample["words"]) - 1
    assert all(b for b, l in zip(sample["breaks"], sample["labels"]) if l == Label.JOIN)


def test_render_matches_model_input():
    sample = corrupt(TEXT, random.Random(0))
    assert split_input(render(sample["words"], sample["breaks"])) == (sample["words"], sample["breaks"])


def test_produces_all_error_types():
    sample = corrupt(TEXT, random.Random(0), width_range=(30, 30), split_prob=1.0)
    pairs = set(zip(sample["breaks"], sample["labels"]))
    assert (True, Label.SPACE) in pairs  # wrong line break to remove
    assert (True, Label.JOIN) in pairs  # broken word to merge


def test_deterministic_with_seed():
    assert corrupt(TEXT, random.Random(1)) == corrupt(TEXT, random.Random(1))
