"""LangChain tools for the AI Math Assistant.

Every math tool takes one natural-ish string (the LLM decides what to pass),
extracts the numbers itself and returns a consistent ``{"result": ...}`` dict.
On bad input the ``result`` is a human-readable error string instead of a number,
so the agent can recover instead of crashing.
"""

from __future__ import annotations

import math
import re
from typing import Union

from langchain_core.tools import tool

from .parsing import NUMBER_PATTERN, extract_numbers

Result = dict[str, Union[float, str]]

NO_NUMBERS = "No numbers found in input."


# --------------------------------------------------------------------------- #
# Plain functions (easy to unit test) ...
# --------------------------------------------------------------------------- #
def add(inputs: str) -> Result:
    numbers = extract_numbers(inputs)
    if not numbers:
        return {"result": NO_NUMBERS}
    return {"result": math.fsum(numbers)}


def subtract(inputs: str) -> Result:
    numbers = extract_numbers(inputs)
    if not numbers:
        return {"result": NO_NUMBERS}
    result = numbers[0]
    for n in numbers[1:]:
        result -= n
    return {"result": result}


def multiply(inputs: str) -> Result:
    numbers = extract_numbers(inputs)
    if not numbers:
        return {"result": NO_NUMBERS}
    return {"result": math.prod(numbers)}


def divide(inputs: str) -> Result:
    numbers = extract_numbers(inputs)
    if not numbers:
        return {"result": NO_NUMBERS}
    if any(n == 0 for n in numbers[1:]):
        return {"result": "Cannot divide by zero."}
    result = numbers[0]
    for n in numbers[1:]:
        result /= n
    return {"result": result}


_POWER_OPERATOR = r"(?:\^|\*\*|to\s+the\s+power\s+of|raised\s+to(?:\s+the\s+power\s+of)?)"
_POWER_RE = re.compile(rf"({NUMBER_PATTERN})\s*{_POWER_OPERATOR}\s*({NUMBER_PATTERN})", re.IGNORECASE)


def calculate_power(input_text: str) -> Result:
    """Calculate x^y from inputs like "2 3", "2, 3", "2^3", "2**3", "2 to the power of 3"."""
    if not isinstance(input_text, str) or not input_text.strip():
        return {"result": "Invalid input. Provide a base and an exponent, e.g. '2 3' or '2^3'."}

    match = _POWER_RE.search(input_text)
    if match:
        base, exponent = float(match.group(1)), float(match.group(2))
    else:
        numbers = extract_numbers(input_text)
        if len(numbers) != 2:
            return {
                "result": "Invalid input. Expected exactly two numbers (base and exponent), "
                f"found {len(numbers)}."
            }
        base, exponent = numbers

    try:
        result = base**exponent
    except ZeroDivisionError:
        return {"result": "Cannot raise 0 to a negative power."}
    except OverflowError:
        return {"result": "Result is too large to compute."}

    if isinstance(result, complex):
        return {"result": "Result is not a real number (negative base with a fractional exponent)."}
    return {"result": result}


# --------------------------------------------------------------------------- #
# ... wrapped as LangChain tools. The docstring is what the LLM reads.
# --------------------------------------------------------------------------- #
@tool
def add_numbers(inputs: str) -> Result:
    """Add all numbers found in the input string, e.g. "10, 20, 30" -> 60.

    Pass the numbers as digits (convert words like "two" to 2 first).
    Returns {"result": <sum>} or {"result": "<error message>"}.
    """
    return add(inputs)


@tool
def subtract_numbers(inputs: str) -> Result:
    """Subtract sequentially: first number minus every following number.

    "100, 20, 10" -> 100 - 20 - 10 = 70. Order matters: for "subtract 50 from 20"
    pass "20, 50". Returns {"result": <difference>} or {"result": "<error message>"}.
    """
    return subtract(inputs)


@tool
def multiply_numbers(inputs: str) -> Result:
    """Multiply all numbers found in the input string, e.g. "2, 3, 4" -> 24.

    Returns {"result": <product>} or {"result": "<error message>"}.
    """
    return multiply(inputs)


@tool
def divide_numbers(inputs: str) -> Result:
    """Divide the first number by each following number, e.g. "100, 5, 2" -> 10.

    Returns {"result": <quotient>} or {"result": "Cannot divide by zero."}.
    """
    return divide(inputs)


@tool
def power_tool(input_text: str) -> Result:
    """Calculate a power: base (x) raised to exponent (y), i.e. x^y.

    Use for exponents, powers, squares and cubes. Pass base first as digits, e.g.
    "2 3", "2^3" or "2 to the power of 3". Convert words first
    ("two to the power of three" -> "2 3"; "5 squared" -> "5 2"; "3 cubed" -> "3 3").
    Returns {"result": <number>} or {"result": "<error message>"}.
    """
    return calculate_power(input_text)


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for factual information (e.g. populations, dates, definitions).

    Returns a short text summary of the most relevant pages.
    """
    try:
        from langchain_community.utilities import WikipediaAPIWrapper
    except ImportError:  # pragma: no cover - optional extra
        return "Wikipedia tool unavailable: install with `pip install .[wiki]`."
    return WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2000).run(query)


MATH_TOOLS = [add_numbers, subtract_numbers, multiply_numbers, divide_numbers, power_tool]
ALL_TOOLS = [*MATH_TOOLS, search_wikipedia]
TOOL_NAMES = [t.name for t in ALL_TOOLS]
