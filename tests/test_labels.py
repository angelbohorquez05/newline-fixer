import pytest

from newline_fixer.labels import Label, align, decode, encode, split_input

TEXT = "3.2.3 Applications of Attention in our Model\n\nThe Transformer uses attention:\n• In layers"


def test_encode_labels():
    words, labels = encode("Title\n\nFirst line.\nSecond line")
    assert words == ["Title", "First", "line.", "Second", "line"]
    assert labels == [Label.PARAGRAPH, Label.SPACE, Label.NEWLINE, Label.SPACE]


def test_roundtrip():
    assert decode(*encode(TEXT)) == TEXT


def test_whitespace_is_normalised():
    assert decode(*encode("  a   b \n \n\n c\t d ")) == "a b\n\nc d"


def test_join_merges_broken_word():
    assert decode(["the", "que", "ries"], [Label.SPACE, Label.JOIN]) == "the queries"


def test_split_input_marks_line_breaks():
    assert split_input("the que\n ries come") == (["the", "que", "ries", "come"], [False, True, False])


def test_empty_text():
    assert encode("") == ([], [])
    assert decode([], []) == ""


def test_align_reads_labels_from_corrected_text():
    broken = "Title The que\nries come\nfrom the encoder."
    fixed = "Title\n\nThe queries come from the encoder."
    words, breaks, labels = align(broken, fixed)
    assert words == ["Title", "The", "que", "ries", "come", "from", "the", "encoder."]
    assert breaks == [False, False, True, False, True, False, False]
    assert labels == [Label.PARAGRAPH, Label.SPACE, Label.JOIN, Label.SPACE, Label.SPACE, Label.SPACE, Label.SPACE]
    assert decode(words, labels) == fixed


def test_align_handles_windows_line_endings():
    assert align("a\r\nb", "a\r\n\r\nb")[2] == [Label.PARAGRAPH]


def test_align_rejects_changed_words():
    with pytest.raises(ValueError):
        align("the quries", "the queries")
