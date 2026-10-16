"""Native HYV3 reasoning/tool protocol, independent of GPU execution."""

import json
import pytest

from rocm_tools.exl3_server.protocol import (
    IncrementalAssistantParser, ProtocolError, parse_assistant_output, reasoning_kwargs,
)
from rocm_tools.exl3_server.schema import prepare_constraints, validate_tool_arguments

TOOLS = [{"type": "function", "function": {"name": "remember", "parameters": {
    "type": "object", "properties": {"text": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["text", "count"], "additionalProperties": False}}}]


@pytest.mark.parametrize("kwargs,mode,thinking", [
    ({}, "no_think", False), ({"reasoning_effort": "no_think"}, "no_think", False),
    ({"reasoning_effort": "high"}, "high", True),
    ({"reasoning_effort": "xhigh"}, "high", True),
    ({"reasoning_effort": "medium"}, "low", True),
    ({"enable_thinking": True}, "low", True),
    ({"enable_thinking": False, "reasoning_effort": "high"}, "no_think", False),
])
def test_native_reasoning_modes(kwargs, mode, thinking):
    result = reasoning_kwargs(kwargs, model_family="hy_v3")
    assert (result["reasoning_effort"], result["enable_thinking"]) == (mode, thinking)
    assert reasoning_kwargs({"reasoning_effort": "high"})["reasoning_effort"] == "xhigh"


def test_native_parallel_calls_stream_without_control_tag_leaks():
    value = '東京: literal <tool_call> and </arg_value>'
    call = ('<tool_call>remember<tool_sep>\n<arg_key>text</arg_key>\n<arg_value>'
            + json.dumps(value, ensure_ascii=False) + '</arg_value>\n'
            '<arg_key>count</arg_key>\n<arg_value>2</arg_value>\n</tool_call>\n')
    payload = '<tool_calls>\n' + call + call + '</tool_calls>'
    parsed = parse_assistant_output(payload, model_family="hy_v3", tools=TOOLS)
    assert parsed["content"] is None and len(parsed["tool_calls"]) == 2
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {"text": value, "count": 2}
    validate_tool_arguments(parsed["tool_calls"], TOOLS)
    parser = IncrementalAssistantParser(model_family="hy_v3", tools=TOOLS)
    events = []
    for byte in payload.encode():
        events += parser.feed(bytes([byte]))
    final = parser.finish()
    events += final["events"]
    assert not [e for e in events if e["type"] == "content"]
    assert len([e for e in events if e["type"] == "tool_call"]) == 2
    assert final["tool_calls"] == parsed["tool_calls"]
    with pytest.raises(ProtocolError, match="incomplete"):
        parse_assistant_output(payload.removesuffix('</tool_calls>'), model_family="hy_v3")


def test_literal_tags_in_json_are_preserved():
    payload = json.dumps({"text": '<think>literal</think> <tool_calls> <｜reasoning_mode｜>'}, ensure_ascii=False)
    parser = IncrementalAssistantParser(model_family="hy_v3")
    events = []
    for char in payload:
        events += parser.feed(char)
    final = parser.finish()
    events += final["events"]
    assert final["content"] == payload and not final["tool_calls"]
    assert ''.join(e['delta'] for e in events if e['type'] == 'content') == payload


def test_native_initial_reasoning_split_close_then_json():
    parser = IncrementalAssistantParser(model_family="hy_v3", initial_reasoning=True)
    events = parser.feed('考えます</thi') + parser.feed('nk>{"memories":[]}')
    final = parser.finish()
    events += final["events"]
    assert final["reasoning_content"] == '考えます'
    assert ''.join(e['delta'] for e in events if e['type'] == 'reasoning_content') == '考えます'
    assert ''.join(e['delta'] for e in events if e['type'] == 'content') == '{"memories":[]}'


def test_native_unfinished_reasoning_never_becomes_a_tool_call():
    thought = '<tool_calls><tool_call>remember<tool_sep></tool_call></tool_calls>'
    partial = parse_assistant_output('<think>' + thought, model_family='hy_v3', allow_partial=True)
    assert not partial['tool_calls'] and partial['content'] is None
    parser = IncrementalAssistantParser(model_family='hy_v3', initial_reasoning=True)
    events = parser.feed(thought)
    final = parser.finish()
    assert not final['tool_calls'] and final['content'] is None
    assert final['reasoning_content'] == thought
    assert all(e['type'] == 'reasoning_content' for e in events + final['events'])


def test_hy_tool_constraints_use_native_group_and_json_values():
    class Tokenizer:
        def single_id(self, text):
            assert text == "<tool_calls>"
            return 120035
    plan = prepare_constraints(Tokenizer(), TOOLS, "required", False, thinking=False,
                               model_family="hy_v3", compile_filters=False)
    assert plan.generation_prefix == "<tool_calls>\n"
    assert "<arg_key>" in plan.filter_spec["llg_grammar"]
    assert "<function=" not in plan.filter_spec["llg_grammar"]
