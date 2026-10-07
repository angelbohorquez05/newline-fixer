"""Strip the Markdown of the source texts so they read like plain document text.

Only markup that would end up as extra words is removed: horizontal rules,
heading marks and bold markers. List bullets become "•", as in most PDFs.
"""

import re

_RULE = re.compile(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", re.M)  # "---", "* * *"
_HEADING = re.compile(r"^[ \t]*#+[ \t]+", re.M)  # "## Title"; wikihow uses up to 7 "#"
_BOLD = re.compile(r"\*\*")  # "**word**"
_BULLET = re.compile(r"^[ \t]*[-*+][ \t]+", re.M)  # "- item", "  * item"


def clean(text: str) -> str:
    text = _RULE.sub("", text)
    text = _HEADING.sub("", text)
    text = _BOLD.sub("", text)  # before bullets, so "**Note**" is not read as a bullet
    text = _BULLET.sub("• ", text)
    return text.strip()
