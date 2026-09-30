"""Command-line interface: `math-assistant "What is 2 to the power of 8?" --trace`."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .agent import PROVIDERS, ask, build_agent, final_answer, get_llm

_USE_COLOR = sys.stdout.isatty() or os.getenv("FORCE_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def print_trace(response: dict) -> None:
    """Pretty-print the agent's message history (the tool-calling loop)."""
    for i, msg in enumerate(response["messages"], start=1):
        if isinstance(msg, SystemMessage):
            continue
        if isinstance(msg, HumanMessage):
            print(f"{_c('1;36', f'[{i}] Human')}      {msg.content}")
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"{_c('1;33', f'[{i}] AI → tool')}  {_c('33', call['name'])}({call['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"{_c('1;35', f'[{i}] Tool')}       {_c('35', msg.name)} → {msg.content}")
        elif isinstance(msg, AIMessage):
            print(f"{_c('1;32', f'[{i}] AI answer')}  {msg.content}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="math-assistant", description="AI Math Assistant (LangChain tool calling)")
    parser.add_argument("question", nargs="*", help="question to ask; omit for interactive mode")
    parser.add_argument("-p", "--provider", choices=PROVIDERS, default=os.getenv("LLM_PROVIDER", "openai"))
    parser.add_argument("-t", "--trace", action="store_true", help="print every message in the tool-calling loop")
    parser.add_argument("-w", "--wiki", action="store_true", help="also give the agent the Wikipedia tool")
    args = parser.parse_args(argv)

    agent = build_agent(get_llm(args.provider), include_wikipedia=args.wiki)
    label = _c("2", f"(provider: {args.provider})")

    def run(question: str) -> None:
        response = ask(agent, question)
        if args.trace:
            print_trace(response)
        else:
            print(_c("1;32", "Answer: ") + final_answer(response))

    if args.question:
        run(" ".join(args.question))
        return 0

    print(f"AI Math Assistant {label} - type 'exit' to quit")
    while True:
        try:
            question = input(_c("1;36", "\n> ")).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            run(question)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
