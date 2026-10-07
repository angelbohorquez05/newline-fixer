"""Non-learned systems the model has to beat. Same interface as the model: (words, breaks) -> labels."""

import re

from newline_fixer.labels import Label

_NUMBERED = re.compile(r"\d+[.)]")  # "1." or "2)" starting a list item
SHORT_LINE = 0.7  # a line below 70% of the text width ended on purpose; tuned on val (D6)


def identity(words: list[str], breaks: list[bool]) -> list[Label]:
    """Return the input unchanged: every line break is kept."""
    return [Label.NEWLINE if b else Label.SPACE for b in breaks]


def vocabulary(texts) -> frozenset[str]:
    """Lower-cased alphabetic words of well-formatted texts, used to detect broken words."""
    return frozenset(w.lower() for text in texts for w in text.split() if w.isalpha())


def _line_lengths(words: list[str], breaks: list[bool]) -> list[int]:
    """Length in characters of the input line each word belongs to."""
    lengths, line = [], []
    for i, word in enumerate(words):
        line.append(word)
        if i == len(breaks) or breaks[i]:
            lengths += [len(" ".join(line))] * len(line)
            line = []
    return lengths


def heuristic(words: list[str], breaks: list[bool], vocab: frozenset[str] = frozenset()) -> list[Label]:
    """Hand-written rules:

    - a bullet "•" always starts a new line;
    - a line break inside a known word ("que" + "ries" = "queries", "ries" unknown) is a JOIN;
    - a line clearly shorter than the text width ended on purpose: NEWLINE before a
      numbered item, PARAGRAPH otherwise;
    - any other line break comes from hard wrapping: SPACE.
    """
    if not words:
        return []
    lengths = _line_lengths(words, breaks)
    width = max(lengths)
    labels = []
    for i, (brk, a, b) in enumerate(zip(breaks, words, words[1:])):
        if b.startswith("•"):
            labels.append(Label.NEWLINE)
        elif not brk:
            labels.append(Label.SPACE)
        elif a.isalpha() and b.isalpha() and (a + b).lower() in vocab and b.lower() not in vocab:
            labels.append(Label.JOIN)
        elif lengths[i] < SHORT_LINE * width:
            labels.append(Label.NEWLINE if _NUMBERED.fullmatch(b) else Label.PARAGRAPH)
        else:
            labels.append(Label.SPACE)
    return labels
