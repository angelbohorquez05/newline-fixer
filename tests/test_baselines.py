from newline_fixer.baselines import heuristic, identity, vocabulary
from newline_fixer.labels import Label, split_input

S, J, N, P = Label.SPACE, Label.JOIN, Label.NEWLINE, Label.PARAGRAPH


def test_identity_keeps_line_breaks():
    assert identity(["a", "b", "c"], [True, False]) == [N, S]


def test_vocabulary_keeps_alphabetic_words():
    assert vocabulary(["The queries, 3 times"]) == {"the", "times"}


def test_heuristic_rules():
    text = (
        "This line is long enough to set the width of the que\n"
        "ries and keeps going on until the very end of it\n"
        "Short heading\n"
        "• a bullet glued to the text"
    )
    words, breaks = split_input(text)
    labels = heuristic(words, breaks, vocab=frozenset({"queries"}))
    assert labels[words.index("que")] == J
    assert labels[words.index("it")] == S  # full-width line: wrapping
    assert labels[words.index("heading")] == N  # bullet follows
    assert labels[words.index("Short") - 1] == S


def test_heuristic_short_line_is_paragraph():
    words, breaks = split_input("A long first line that sets the width\nEnd.\nNext paragraph here")
    assert heuristic(words, breaks)[words.index("End.")] == P


def test_heuristic_empty():
    assert heuristic([], []) == []
