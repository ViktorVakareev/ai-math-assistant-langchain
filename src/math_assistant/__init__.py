"""AI Math Assistant - a LangChain tool-calling agent for arithmetic."""

from .agent import SYSTEM_PROMPT, ask, build_agent, final_answer, get_llm, last_tool_result, tool_calls
from .tools import (
    ALL_TOOLS,
    MATH_TOOLS,
    add_numbers,
    calculate_power,
    divide_numbers,
    multiply_numbers,
    power_tool,
    search_wikipedia,
    subtract_numbers,
)

__version__ = "1.0.0"

__all__ = [
    "ALL_TOOLS", "MATH_TOOLS", "SYSTEM_PROMPT", "add_numbers", "ask", "build_agent",
    "calculate_power", "divide_numbers", "final_answer", "get_llm", "last_tool_result",
    "multiply_numbers", "power_tool", "search_wikipedia", "subtract_numbers", "tool_calls",
]
