import pytest

from newline_fixer.labels import Label
from newline_fixer.metrics import evaluate

S, J, N, P = Label.SPACE, Label.JOIN, Label.NEWLINE, Label.PARAGRAPH


def test_perfect_prediction():
    gold = [[S, J, N], [P, S]]
    result = evaluate(gold, gold)
    assert result["macro_f1"] == result["exact_match"] == 1.0


def test_counts_per_label():
    result = evaluate([[S, S, N, N]], [[S, N, N, S]])
    assert result["labels"]["NEWLINE"] == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert result["labels"]["JOIN"]["f1"] == 0.0  # never present, never predicted
    assert result["exact_match"] == 0.0


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        evaluate([[S, S]], [[S]])
