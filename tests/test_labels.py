from newline_fixer.labels import Label, decode, encode, split_input

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
