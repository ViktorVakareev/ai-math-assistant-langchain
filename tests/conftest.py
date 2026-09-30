import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class ScriptedToolModel(GenericFakeChatModel):
    """Fake chat model that replays scripted AIMessages (incl. tool calls)."""

    def bind_tools(self, tools, **kwargs):
        return self


def tool_call(name, text, call_id="call_1"):
    arg = "input_text" if name == "power_tool" else "inputs"
    return AIMessage(content="", tool_calls=[{"name": name, "args": {arg: text}, "id": call_id}])


@pytest.fixture
def scripted_model():
    def make(*messages):
        return ScriptedToolModel(messages=iter(messages))

    return make
