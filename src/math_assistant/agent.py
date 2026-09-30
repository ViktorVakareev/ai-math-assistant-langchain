"""Agent factory and helpers for the AI Math Assistant."""

from __future__ import annotations

import json
import os
from typing import Any, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import ToolMessage

from .tools import ALL_TOOLS, MATH_TOOLS

SYSTEM_PROMPT = (
    "You are a helpful mathematical assistant. Always use the provided tools for "
    "arithmetic instead of calculating yourself. Pass numbers as digits, convert number "
    "words first, and break multi-step problems into one tool call per step. "
    "Explain your reasoning briefly and state the final result clearly."
)

PROVIDERS = ("openai", "watsonx", "offline")


def get_llm(provider: Optional[str] = None, **kwargs: Any) -> BaseChatModel:
    """Build a chat model. ``provider`` defaults to $LLM_PROVIDER or "openai".

    - openai:  needs OPENAI_API_KEY (model: $OPENAI_MODEL, default gpt-5.4-nano)
    - watsonx: needs WATSONX_APIKEY and WATSONX_PROJECT_ID (model: $WATSONX_MODEL)
    - offline: rule-based stand-in, no key/network (see ``offline.py``)
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "openai")).lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-5.4-nano"), **kwargs)

    if provider == "watsonx":
        from langchain_ibm import ChatWatsonx

        return ChatWatsonx(
            model_id=os.getenv("WATSONX_MODEL", "ibm/granite-4-h-small"),
            url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
            project_id=os.environ["WATSONX_PROJECT_ID"],
            **kwargs,
        )

    if provider == "offline":
        from .offline import OfflineMathModel

        return OfflineMathModel()

    raise ValueError(f"Unknown provider {provider!r}. Choose one of {PROVIDERS}.")


def build_agent(llm: Optional[BaseChatModel] = None, tools: Optional[Sequence] = None,
                include_wikipedia: bool = False, system_prompt: str = SYSTEM_PROMPT):
    """Create the tool-calling agent (a LangGraph graph: model -> tools -> model ...)."""
    if tools is None:
        tools = ALL_TOOLS if include_wikipedia else MATH_TOOLS
    return create_agent(model=llm or get_llm(), tools=list(tools), system_prompt=system_prompt)


def ask(agent, question: str) -> dict:
    """Invoke the agent with one human message and return the full response state."""
    return agent.invoke({"messages": [("human", question)]})


def final_answer(response: dict) -> str:
    return str(response["messages"][-1].content)


def tool_calls(response: dict) -> list[dict]:
    """All tool calls the model requested, in order: [{"name", "args", "result"}, ...]."""
    results = {m.tool_call_id: m for m in response["messages"] if isinstance(m, ToolMessage)}
    calls = []
    for msg in response["messages"]:
        for call in getattr(msg, "tool_calls", None) or []:
            tool_msg = results.get(call["id"])
            calls.append({"name": call["name"], "args": call["args"],
                          "result": _parse_result(tool_msg.content) if tool_msg else None})
    return calls


def last_tool_result(response: dict, tool_name: Optional[str] = None) -> Any:
    """Result of the last (optionally: named) tool call, or None if no tool ran."""
    calls = [c for c in tool_calls(response) if tool_name in (None, c["name"])]
    return calls[-1]["result"] if calls else None


def _parse_result(content: Any) -> Any:
    try:
        return json.loads(content)["result"]
    except (TypeError, ValueError, KeyError):
        return content
