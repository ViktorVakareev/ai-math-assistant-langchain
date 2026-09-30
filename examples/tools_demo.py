"""Inspect and call the tools directly - no LLM involved.

    python examples/tools_demo.py
"""

from math_assistant import MATH_TOOLS

print("Registered tools (what the LLM sees):")
for t in MATH_TOOLS:
    print(f"  • {t.name:17} args={list(t.args)}")
    print(f"    {t.description.splitlines()[0]}")

print("\nDirect calls:")
calls = [
    ("add_numbers", "10 20 30 a b"),
    ("subtract_numbers", "100, 20, 10"),
    ("multiply_numbers", "2, 3, and 4"),
    ("divide_numbers", "100, 5, 2"),
    ("divide_numbers", "10, 0"),
    ("power_tool", "2 to the power of 8"),
    ("power_tool", "1.5 ^ 2"),
    ("power_tool", "0 -1"),
]
tools = {t.name: t for t in MATH_TOOLS}
for name, text in calls:
    call = f"{name}.invoke({text!r})"
    print(f"  {call:44} -> {tools[name].invoke(text)}")
