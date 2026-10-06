"""Convert between text and (words, gap labels).

The task is framed as classifying every gap between two consecutive words.
Words are never modified; only the separator between them is predicted.
"""

import re
from enum import IntEnum

_WORD = re.compile(r"\S+")


class Label(IntEnum):
    SPACE = 0  # "a b"
    JOIN = 1  # "a" + "b" -> "ab": a word broken across lines
    NEWLINE = 2  # "a\nb"
    PARAGRAPH = 3  # "a\n\nb"


SEPARATOR = {Label.SPACE: " ", Label.JOIN: "", Label.NEWLINE: "\n", Label.PARAGRAPH: "\n\n"}


def _split(text: str) -> tuple[list[str], list[str]]:
    """Return the words and the whitespace found between each consecutive pair."""
    matches = list(_WORD.finditer(text))
    words = [m.group() for m in matches]
    gaps = [text[a.end() : b.start()] for a, b in zip(matches, matches[1:])]
    return words, gaps


def encode(text: str) -> tuple[list[str], list[Label]]:
    """Gold labels of a well-formatted text (it never contains JOIN)."""
    words, gaps = _split(text)
    labels = [
        Label.PARAGRAPH if g.count("\n") >= 2 else Label.NEWLINE if "\n" in g else Label.SPACE
        for g in gaps
    ]
    return words, labels


def split_input(text: str) -> tuple[list[str], list[bool]]:
    """Model input: the words and whether each gap contained a line break."""
    words, gaps = _split(text)
    return words, ["\n" in g for g in gaps]


def decode(words: list[str], labels: list[Label]) -> str:
    """Rebuild the text from words and one label per gap."""
    if not words:
        return ""
    return words[0] + "".join(SEPARATOR[Label(l)] + w for l, w in zip(labels, words[1:]))
