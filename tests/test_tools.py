"""Unit tests for the math tools - deterministic, no LLM involved."""

import math

import pytest

from math_assistant.parsing import extract_numbers
from math_assistant.tools import (
    MATH_TOOLS,
    add_numbers,
    calculate_power,
    divide_numbers,
    multiply_numbers,
    power_tool,
    subtract_numbers,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Add 10, 20 and 30", [10, 20, 30]),
        ("10 20 30 a b", [10, 20, 30]),
        ("1.5 and -2.25", [1.5, -2.25]),
        ("no numbers here", []),
        ("", []),
    ],
)
def test_extract_numbers(text, expected):
    assert extract_numbers(text) == expected


class TestAdd:
    @pytest.mark.parametrize(
        "text, expected",
        [("10, 20, 30", 60), ("10 20 30 a b", 60), ("27.72, 2.14, 1.79", 31.65), ("-10, -20, -30", -60)],
    )
    def test_sums(self, text, expected):
        assert math.isclose(add_numbers.invoke(text)["result"], expected)

    def test_no_numbers(self):
        assert add_numbers.invoke("nothing")["result"] == "No numbers found in input."


class TestSubtract:
    @pytest.mark.parametrize(
        "text, expected",
        [("100, 20, 10", 70), ("20, 50", -30), ("1.5 0.5", 1.0), ("-5 5", -10)],
    )
    def test_sequential(self, text, expected):
        assert subtract_numbers.invoke(text)["result"] == expected

    def test_no_numbers(self):
        assert subtract_numbers.invoke("")["result"] == "No numbers found in input."


class TestMultiply:
    @pytest.mark.parametrize("text, expected", [("2, 3, 4", 24), ("3.5 2", 7), ("-2 3", -6), ("5 0", 0)])
    def test_product(self, text, expected):
        assert multiply_numbers.invoke(text)["result"] == expected


class TestDivide:
    @pytest.mark.parametrize("text, expected", [("100, 5, 2", 10), ("25 4", 6.25), ("-9 3", -3)])
    def test_sequential(self, text, expected):
        assert divide_numbers.invoke(text)["result"] == expected

    def test_divide_by_zero(self):
        assert divide_numbers.invoke("10, 0")["result"] == "Cannot divide by zero."


class TestPower:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("2 3", 8), ("2, 3", 8), ("5^2", 25), ("2**10", 1024),
            ("Calculate 5 to the power of 2.", 25), ("3 raised to the power of 4", 81),
            ("1.5 ^ 2", 2.25), ("-2 3", -8), ("2 -1", 0.5),
        ],
    )
    def test_valid(self, text, expected):
        assert math.isclose(power_tool.invoke(text)["result"], expected)

    @pytest.mark.parametrize(
        "text, message",
        [
            ("0 -1", "Cannot raise 0 to a negative power."),
            ("-8 0.5", "not a real number"),
            ("2", "found 1"),
            ("two to the power of three", "found 0"),
            ("", "Provide a base and an exponent"),
            ("10 10000", "too large"),
        ],
    )
    def test_errors_are_returned_not_raised(self, text, message):
        assert message in calculate_power(text)["result"]


def test_tools_expose_names_descriptions_and_schema():
    for t in MATH_TOOLS:
        assert t.name and len(t.description) > 20
        assert len(t.args) == 1  # single string input
    assert set(power_tool.args) == {"input_text"}
