"""A deterministic, rule-based stand-in for an LLM - no API key, no network.

``OfflineMathModel`` speaks the same tool-calling protocol as a real chat model
(it emits ``AIMessage.tool_calls`` and reads ``ToolMessage`` results), so the real
LangChain agent loop, tools and message history run exactly as they would with
OpenAI or watsonx. Only the "reasoning" is replaced by keyword rules.

Use it for demos, screenshots and fast offline tests. It is *not* an LLM: it only
understands simple arithmetic phrasing, optionally chained with "then".
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .parsing import NUMBER_PATTERN

_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}
_WORD_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")\b", re.IGNORECASE)
_CLAUSE_SPLIT = re.compile(r"\s*(?:,\s*)?\b(?:and\s+)?then\b\s*", re.IGNORECASE)

# (tool name, argument name, keyword regex) - checked in order
_ROUTES = [
    ("power_tool", "input_text", r"power|squared|cubed|exponent|\^|\*\*"),
    ("divide_numbers", "inputs", r"divide|divided|quotient|\s/\s"),
    ("multiply_numbers", "inputs", r"multiply|times|product"),
    ("subtract_numbers", "inputs", r"subtract|minus|difference|take away"),
    ("add_numbers", "inputs", r"\badd\b|plus|sum|total"),
]


def _fmt(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _words_to_digits(text: str) -> str:
    return _WORD_RE.sub(lambda m: str(_WORDS[m.group(1).lower()]), text)


def plan_tool_call(clause: str, previous: Optional[float] = None,
                   inherit: Optional[str] = None) -> Optional[tuple[str, dict]]:
    """Map one clause of the user query to (tool_name, args), or None if not understood.

    ``inherit`` is the previous step's tool, reused for clauses like "... then by 2".
    """
    text = _words_to_digits(clause)
    for name, arg, pattern in _ROUTES:
        if re.search(pattern, text, re.IGNORECASE) or (
            inherit == name and not any(re.search(p, text, re.IGNORECASE) for _, _, p in _ROUTES)
        ):
            numbers = re.findall(NUMBER_PATTERN, text)
            if name == "power_tool":
                if re.search(r"squared", text, re.IGNORECASE):
                    numbers.append("2")
                elif re.search(r"cubed", text, re.IGNORECASE):
                    numbers.append("3")
            if name == "subtract_numbers" and re.search(r"\bfrom\b", text) and len(numbers) == 2:
                numbers.reverse()  # "subtract 50 from 20" -> 20 - 50
            if previous is not None:
                numbers.insert(0, _fmt(previous))
            if not numbers:
                return None
            return name, {arg: ", ".join(numbers)}
    return None


class OfflineMathModel(BaseChatModel):
    """Rule-based chat model that drives the math tools through real tool calls."""

    @property
    def _llm_type(self) -> str:
        return "offline-math-rules"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "OfflineMathModel":
        return self

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        # Work only on the latest human turn
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        query = str(messages[last_human].content)
        tool_results = [m for m in messages[last_human:] if isinstance(m, ToolMessage)]

        clauses = [c for c in _CLAUSE_SPLIT.split(query.strip().rstrip(".?!")) if c]
        steps_done = len(tool_results)
        previous = None
        if tool_results:
            previous = json.loads(tool_results[-1].content).get("result")
            if isinstance(previous, str):  # the tool reported an error -> stop and report it
                return self._answer(f"The tool reported: {previous}")

        if steps_done < len(clauses):
            inherit = tool_results[-1].name if tool_results else None
            plan = plan_tool_call(clauses[steps_done], previous, inherit)
            if plan:
                name, args = plan
                call = {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])
            if steps_done == 0:
                return self._answer("Offline demo model: I can only handle simple arithmetic "
                                    "(add, subtract, multiply, divide, powers).")

        return self._answer(f"The result is {_fmt(previous)}.")

    @staticmethod
    def _answer(text: str) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
