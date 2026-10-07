import pytest

from newline_fixer.clean import clean


@pytest.mark.parametrize(
    "markdown, expected",
    [
        ("# Title\n\n## Section 1\nText", "Title\n\nSection 1\nText"),
        ("####### Step 6: Soak\nText", "Step 6: Soak\nText"),
        ("Use **bold** words", "Use bold words"),
        ("**Step 1: Mix**\nThen wait", "Step 1: Mix\nThen wait"),
        ("Items:\n- one\n* two\n  + three", "Items:\n• one\n• two\n• three"),
        ("* **Tip:** rest", "• Tip: rest"),
        ("Part one\n\n---\n\nPart two", "Part one\n\n\n\nPart two"),
        ("1. First\n2. Second", "1. First\n2. Second"),
    ],
)
def test_clean(markdown, expected):
    assert clean(markdown) == expected


def test_keeps_inline_symbols():
    text = "Use C# and 3 * 4 = 12, a - b, #hashtag"
    assert clean(text) == text
