"""Multi-step: look a fact up on Wikipedia, then compute with it (needs a real LLM + network).

    pip install -e ".[openai,wiki]"
    python examples/wikipedia_plus_math.py
"""

from dotenv import load_dotenv

from math_assistant import ask, build_agent, final_answer, get_llm, tool_calls

load_dotenv()
agent = build_agent(get_llm(), include_wikipedia=True)
response = ask(agent, "What is the population of Canada? Multiply it by 0.75")
for call in tool_calls(response):
    print(f"tool: {call['name']}({call['args']})")
print(final_answer(response))
