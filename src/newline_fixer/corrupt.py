"""Corrupt well-formatted text the way PDF extraction and hard wrapping do.

The output keeps the gold label of every gap, so (input, target) pairs are free.
"""

import random

from newline_fixer.labels import Label, encode


def corrupt(
    text: str,
    rng: random.Random,
    width_range: tuple[int, int] = (40, 100),
    split_prob: float = 0.3,
    keep_break_prob: float = 0.5,
) -> dict:
    """Return {"words", "breaks", "labels"}; breaks[i] and labels[i] describe the gap after words[i].

    - Lines are hard-wrapped at a random width, adding wrong line breaks (gold SPACE).
    - With `split_prob`, the word overflowing the line is cut in two (gold JOIN).
    - Real line/paragraph breaks are kept as a single break with `keep_break_prob`,
      otherwise glued to the next word (e.g. "Model The Transformer").
    """
    words, gold = encode(text)
    if not words:
        return {"words": [], "breaks": [], "labels": []}

    width = rng.randint(*width_range)
    out, breaks, labels = [words[0]], [], []
    line = len(words[0])
    for word, label in zip(words[1:], gold):
        if label != Label.SPACE:
            brk = rng.random() < keep_break_prob
        else:
            brk = line + 1 + len(word) > width
            if brk and len(word) >= 6 and word.isalpha() and rng.random() < split_prob:
                cut = rng.randint(2, len(word) - 2)
                out.append(word[:cut])
                breaks.append(False)
                labels.append(Label.SPACE)
                word, label = word[cut:], Label.JOIN
        out.append(word)
        breaks.append(brk)
        labels.append(label)
        line = len(word) if brk else line + 1 + len(word)

    return {"words": out, "breaks": breaks, "labels": [int(l) for l in labels]}


def render(words: list[str], breaks: list[bool]) -> str:
    """The corrupted text as a user would paste it."""
    if not words:
        return ""
    return words[0] + "".join(("\n" if b else " ") + w for b, w in zip(breaks, words[1:]))
