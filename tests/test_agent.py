"""Agent tests: the real create_agent loop and real tools, with a fake/offline model.

These check the *orchestration* (tool selection, message flow, chaining, result
extraction) deterministically - no API key needed.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from conftest import tool_call
from math_assistant import ask, build_agent, final_answer, last_tool_result, tool_calls
from math_assistant.agent import get_llm
from math_assistant.offline import OfflineMathModel, plan_tool_call


def test_agent_runs_tool_and_returns_final_answer(scripted_model):
    llm = scripted_model(tool_call("power_tool", "5 2"), AIMessage(content="5 to the power of 2 is 25."))
    response = ask(build_agent(llm), "Calculate 5 to the power of 2.")

    types = [type(m) for m in response["messages"]]
    assert types == [HumanMessage, AIMessage, ToolMessage, AIMessage]
    assert last_tool_result(response, "power_tool") == 25
    assert final_answer(response) == "5 to the power of 2 is 25."


def test_agent_chains_multiple_tools(scripted_model):
    llm = scripted_model(
        tool_call("add_numbers", "25, 15", "c1"),
        tool_call("multiply_numbers", "40, 2", "c2"),
        AIMessage(content="(25 + 15) * 2 = 80"),
    )
    response = ask(build_agent(llm), "Add 25 and 15, then multiply by 2")
    assert [(c["name"], c["result"]) for c in tool_calls(response)] == [
        ("add_numbers", 40), ("multiply_numbers", 80)]


def test_tool_error_is_passed_back_to_model(scripted_model):
    llm = scripted_model(tool_call("divide_numbers", "10, 0"), AIMessage(content="You can't divide by zero."))
    response = ask(build_agent(llm), "Divide 10 by 0")
    assert last_tool_result(response) == "Cannot divide by zero."


def test_no_tool_call_returns_none(scripted_model):
    response = ask(build_agent(scripted_model(AIMessage(content="Hello!"))), "Hi")
    assert last_tool_result(response) is None and tool_calls(response) == []


# ---- the offline rule-based model used for demos --------------------------------
@pytest.mark.parametrize(
    "query, expected_calls",
    [
        ("What is 25 divided by 4?", [("divide_numbers", 6.25)]),
        ("Subtract 100, 20, and 10.", [("subtract_numbers", 70)]),
        ("Subtract 50 from 20.", [("subtract_numbers", -30)]),
        ("Multiply 2, 3, and 4.", [("multiply_numbers", 24)]),
        ("Divide 100 by 5 and then by 2.", [("divide_numbers", 20), ("divide_numbers", 10)]),
        ("Add 10, 20, two and 30", [("add_numbers", 62)]),
        ("What is 12 squared?", [("power_tool", 144)]),
        ("Add 25 and 15, then multiply by 2", [("add_numbers", 40), ("multiply_numbers", 80)]),
        ("What is 2 to the power of 8, then subtract 6?", [("power_tool", 256), ("subtract_numbers", 250)]),
    ],
)
def test_offline_model_end_to_end(query, expected_calls):
    response = ask(build_agent(OfflineMathModel()), query)
    assert [(c["name"], c["result"]) for c in tool_calls(response)] == expected_calls
    assert final_answer(response).startswith("The result is")


def test_offline_model_stops_on_tool_error():
    response = ask(build_agent(OfflineMathModel()), "Divide 10 by 0")
    assert final_answer(response) == "The tool reported: Cannot divide by zero."


def test_offline_model_declines_non_math():
    response = ask(build_agent(OfflineMathModel()), "Who wrote Hamlet?")
    assert tool_calls(response) == [] and "only handle simple arithmetic" in final_answer(response)


def test_plan_tool_call_uses_previous_result():
    assert plan_tool_call("multiply by 2", previous=40.0) == ("multiply_numbers", {"inputs": "40, 2"})


def test_get_llm_offline_and_unknown_provider():
    assert isinstance(get_llm("offline"), OfflineMathModel)
    with pytest.raises(ValueError):
        get_llm("nope")
