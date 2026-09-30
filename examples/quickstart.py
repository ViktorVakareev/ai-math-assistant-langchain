"""Quickstart: build the agent, ask questions, inspect the tool calls.

    python examples/quickstart.py            # uses LLM_PROVIDER from .env (default: openai)
    python examples/quickstart.py offline    # no API key needed
"""

import sys

from dotenv import load_dotenv

from math_assistant import ask, build_agent, final_answer, get_llm, tool_calls

load_dotenv()
provider = sys.argv[1] if len(sys.argv) > 1 else None
agent = build_agent(get_llm(provider))

for question in [
    "What is 25 divided by 4?",
    "Add 25 and 15, then multiply by 2",
    "Calculate 5 to the power of 2.",
]:
    response = ask(agent, question)
    print(f"\nQ: {question}")
    for call in tool_calls(response):
        print(f"   tool: {call['name']}({call['args']}) -> {call['result']}")
    print(f"A: {final_answer(response)}")
