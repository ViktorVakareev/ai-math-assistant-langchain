"""Generate (and execute) notebooks/walkthrough.ipynb - run: python scripts/build_notebook.py"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# AI Math Assistant — walkthrough\n\n"
       "Tool calling with LangChain: `@tool` functions + `create_agent`.\n"
       "Runs **offline by default** (rule-based stand-in model, no API key). "
       "Set `PROVIDER = \"openai\"` or `\"watsonx\"` to use a real LLM."),
    code("PROVIDER = \"offline\"  # \"openai\" | \"watsonx\" | \"offline\"\n\n"
         "from dotenv import load_dotenv\nload_dotenv()\n\n"
         "from math_assistant import (MATH_TOOLS, power_tool, add_numbers, divide_numbers,\n"
         "                            build_agent, get_llm, ask, final_answer, tool_calls)\n"
         "from math_assistant.cli import print_trace"),
    md("## 1. Tools are plain functions with a name, a description and a schema\n"
       "The LLM only sees the name, the docstring and the argument schema — that's how it decides which tool to call."),
    code("for t in MATH_TOOLS:\n    print(f\"{t.name:18} args={list(t.args)}  | {t.description.splitlines()[0]}\")"),
    md("## 2. Test tools directly before giving them to an agent"),
    code("print(add_numbers.invoke(\"10 20 30 a b\"))\n"
         "print(divide_numbers.invoke(\"100, 5, 2\"))\n"
         "print(divide_numbers.invoke(\"10, 0\"))          # errors are returned, not raised\n"
         "print(power_tool.invoke(\"2 to the power of 8\"))"),
    md("## 3. Build the agent and ask a question"),
    code("agent = build_agent(get_llm(PROVIDER))\n"
         "response = ask(agent, \"What is 25 divided by 4?\")\n"
         "final_answer(response)"),
    md("## 4. Look inside the loop: Human → AI tool call → Tool result → AI answer"),
    code("print_trace(ask(agent, \"Add 25 and 15, then multiply by 2\"))"),
    md("## 5. Evaluate: assert on the tool result, not the wording"),
    code("cases = [(\"Subtract 100, 20, and 10.\", 70), (\"Multiply 2, 3, and 4.\", 24),\n"
         "         (\"Divide 100 by 5 and then by 2.\", 10), (\"Subtract 50 from 20.\", -30),\n"
         "         (\"Calculate 5 to the power of 2.\", 25), (\"What is 12 squared?\", 144)]\n\n"
         "passed = 0\n"
         "for query, expected in cases:\n"
         "    calls = tool_calls(ask(agent, query))\n"
         "    result = calls[-1][\"result\"] if calls else None\n"
         "    ok = result == expected\n"
         "    passed += ok\n"
         "    print(f\"{'✅' if ok else '❌'} {query:35} -> {result}\")\n"
         "print(f\"\\n{passed}/{len(cases)} passed\")"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "walkthrough.ipynb"
NotebookClient(nb, timeout=120, kernel_name="python3").execute()
nbf.write(nb, out)
print("wrote", out)
