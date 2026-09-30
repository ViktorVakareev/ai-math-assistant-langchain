"""Shared input parsing for the math tools."""

from __future__ import annotations

import re

NUMBER_PATTERN = r"-?\d+(?:\.\d+)?"
_NUMBER_RE = re.compile(NUMBER_PATTERN)


def extract_numbers(text: str) -> list[float]:
    """Return every integer/decimal (including negatives) found in ``text``, in order.

    >>> extract_numbers("Add 10, 20.5 and -3")
    [10.0, 20.5, -3.0]
    """
    if not isinstance(text, str):
        return []
    return [float(n) for n in _NUMBER_RE.findall(text)]
