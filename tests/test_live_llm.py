"""Live evaluation against a real LLM (the lab's test cases + the power exercise).

Skipped unless OPENAI_API_KEY is set. Run with:  pytest -m live -v
Asserts on the *tool result*, not the model's wording, so it's robust to phrasing.
"""

import math
import os

import pytest

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]

CASES = [
    ("Subtract 100, 20, and 10.", 70),
    ("Multiply 2, 3, and 4.", 24),
    ("Divide 100 by 5 and then by 2.", 10),
    ("Subtract 50 from 20.", -30),
    ("Calculate 5 to the power of 2.", 25),
    ("What is two to the power of three?", 8),
    ("What is 3 cubed?", 27),
    ("Add 25 and 15, then multiply by 2", 80),
]


@pytest.fixture(scope="module")
def agent():
    from math_assistant import build_agent, get_llm

    return build_agent(get_llm("openai"))


@pytest.mark.parametrize("query, expected", CASES)
def test_live_agent(agent, query, expected):
    from math_assistant import ask, last_tool_result

    result = last_tool_result(ask(agent, query))
    assert result is not None, "agent did not call a tool"
    assert math.isclose(float(result), expected)
