#!/usr/bin/env python3
"""CPU-only protocol tests; no server/model/native imports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from rocm_tools.exl3_server.protocol import (  # noqa: E402
    IncrementalAssistantParser,
    ProtocolError,
    build_chat_response,
    event_to_chat_delta,
    messages_for_template,
    normalize_chat_messages,
    parse_assistant_output,
    validate_parallel_tool_calls,
    validate_tool_choice,
)


TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}}]


def test_qwen_reasoning_and_function_xml_parse_without_reasoning_tool_lookalike():
    text = (
        "<think>consider <tool_call><function=decoy></function></tool_call></think>\n"
        "Visible answer\n\n<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\"Tokyo\"</parameter>\n"
        "<parameter=units>metric</parameter>\n</function>\n</tool_call>"
    )
    parsed = parse_assistant_output(text)
    assert parsed["reasoning_content"].startswith("consider")
    assert parsed["content"] == "Visible answer"
    assert len(parsed["tool_calls"]) == 1
    call = parsed["tool_calls"][0]
    assert call["id"] == "call_0"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Tokyo", "units": "metric"}
    assert parsed["finish_reason"] == "tool_calls"


def test_history_preserves_tool_ids_reasoning_and_json_arguments():
    messages = normalize_chat_messages([
        {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
        {"role": "assistant", "content": None, "reasoning_content": "brief",
         "tool_calls": [{"id": "call_9", "type": "function",
                          "function": {"name": "get_weather", "arguments": {"city": "Osaka"}}}]},
        {"role": "tool", "tool_call_id": "call_9", "content": "sunny"},
    ])
    assert messages[0]["content"] == "weather?"
    assert messages[1]["reasoning_content"] == "brief"
    assert messages[1]["tool_calls"][0]["id"] == "call_9"
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {"city": "Osaka"}
    assert messages[2]["tool_call_id"] == "call_9"
    templ = messages_for_template(messages, TOOLS)
    assert isinstance(templ[1]["tool_calls"][0]["function"]["arguments"], dict)
    with pytest.raises(ProtocolError):
        normalize_chat_messages([{"role": "tool", "tool_call_id": "missing", "content": "x"}])


def test_retired_tool_remains_valid_history_but_cannot_be_called_now():
    history = [{"role": "user", "content": "run"},
               {"role": "assistant", "content": None, "tool_calls": [{
                   "id": "once", "type": "function", "function": {
                       "name": "retired_tool", "arguments": '{"value":"kept"}'}}]},
               {"role": "tool", "tool_call_id": "once", "content": "done"}]
    adapted = messages_for_template(history, TOOLS)
    assert adapted[1]["tool_calls"][0]["function"]["arguments"] == {"value": "kept"}
    assert adapted[2]["tool_call_id"] == "once"
    from rocm_tools.exl3_server.protocol import validate_output_tools
    with pytest.raises(ProtocolError, match="undeclared"):
        validate_output_tools({"tool_calls": history[1]["tool_calls"]}, TOOLS)


def test_qwen_template_mode_quotes_multiline_strings_but_keeps_typed_values():
    tools = [{"type": "function", "function": {
        "name": "write", "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "count": {"type": "integer"}}}}}]
    messages = [{"role": "user", "content": "write it"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_keep", "type": "function", "function": {
                        "name": "write",
                        "arguments": json.dumps({"text": "line1\n東京", "count": 7},
                                                  ensure_ascii=False)}}]}]
    templ = messages_for_template(messages, tools, json_parameter_values=True)
    args = templ[1]["tool_calls"][0]["function"]["arguments"]
    assert templ[1]["tool_calls"][0]["id"] == "call_keep"
    assert args["text"] == '"line1\\n東京"'
    assert args["count"] == 7
    # The transformed values are valid Qwen parameter source and recover the
    # same semantic arguments through the normal assistant-output parser.
    rendered = ('<tool_call><function=write><parameter=text>\n' + args["text"]
                + '\n</parameter><parameter=count>\n' + str(args["count"])
                + '\n</parameter></function></tool_call>')
    parsed = parse_assistant_output(rendered, tools=tools)
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {
        "text": "line1\n東京", "count": 7}
    assert json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]) == {
        "text": "line1\n東京", "count": 7}


def test_tool_choice_and_parallel_validation():
    assert validate_tool_choice("auto", TOOLS) == "auto"
    assert validate_tool_choice("none", TOOLS) == "none"
    assert validate_tool_choice("required", TOOLS) == "required"
    assert validate_tool_choice({"type": "function", "function": {"name": "get_weather"}}, TOOLS)
    with pytest.raises(ProtocolError):
        validate_tool_choice({"type": "function", "function": {"name": "missing"}}, TOOLS)
    with pytest.raises(ProtocolError):
        validate_parallel_tool_calls([{"id": "a"}, {"id": "b"}], parallel_tool_calls=False)


def test_incremental_parser_handles_split_utf8_escaped_json_and_reasoning():
    raw = '<think>slow</think>\n答え\n<tool_call><function=get_weather><parameter=city>"東京\\n駅"</parameter></function></tool_call>'
    data = raw.encode("utf-8")
    parser = IncrementalAssistantParser()
    events = []
    # Split both a UTF-8 sequence and escaped JSON text across chunks.
    for i in range(0, len(data), 3):
        events.extend(parser.feed(data[i:i + 3]))
    final = parser.finish()
    events.extend(final["events"])
    assert any(e["type"] == "reasoning_content" and e["delta"] == "slow" for e in events)
    assert "答え" in "".join(e["delta"] for e in events if e["type"] == "content")
    starts = [e for e in events if e["type"] == "tool_call_start"]
    args = "".join(e["delta"] for e in events if e["type"] == "tool_call_arguments")
    assert len(starts) == 1
    assert starts[0]["name"] == "get_weather"
    assert json.loads(args) == {"city": "東京\n駅"}
    assert final["finish_reason"] == "tool_calls"


def test_incremental_quoted_json_source_matches_full_args_for_code_string():
    value = '"""docstring"""\nquoted "inner" and \\ path\\'
    raw_value = json.dumps(value, ensure_ascii=False)
    raw = ('<tool_call><function=get_weather><parameter=city>\n' + raw_value
           + '\n</parameter></function></tool_call>')
    full = parse_assistant_output(raw)
    full_args = full["tool_calls"][0]["function"]["arguments"]
    parser = IncrementalAssistantParser(tools=TOOLS)
    events = []
    for char in raw:
        events.extend(parser.feed(char))
    events.extend(parser.finish()["events"])
    streamed_args = "".join(e["delta"] for e in events
                              if e["type"] == "tool_call_arguments")
    assert streamed_args == full_args
    assert json.loads(streamed_args) == {"city": value}


def test_tool_start_after_unbalanced_prose_quote_is_not_hidden_by_quote_scanner():
    prose = 'The explanation mentions an escaped quote: \\" before the call.\n'
    raw = prose + (
        '<tool_call><function=get_weather><parameter=city>\n"Tokyo"\n'
        '</parameter></function></tool_call>'
    )
    full = parse_assistant_output(raw, tools=TOOLS)
    parser = IncrementalAssistantParser(tools=TOOLS)
    events = []
    for char in raw:
        events.extend(parser.feed(char))
    events.extend(parser.finish()["events"])
    streamed_args = "".join(e["delta"] for e in events
                              if e["type"] == "tool_call_arguments")
    assert len(full["tool_calls"]) == 1
    assert streamed_args == full["tool_calls"][0]["function"]["arguments"]
    assert json.loads(streamed_args) == {"city": "Tokyo"}


def test_partial_tool_header_is_buffered_without_xml_content_leak():
    raw = ('<tool_call><function=get_weather><parameter=city>"Tokyo"'
           '</parameter></function></tool_call>')
    parser = IncrementalAssistantParser(tools=TOOLS)
    events = []
    for char in raw:
        events.extend(parser.feed(char))
    events.extend(parser.finish()["events"])
    content = "".join(e.get("delta", "") for e in events if e["type"] == "content")
    starts = [e for e in events if e["type"] == "tool_call_start"]
    args = "".join(e["delta"] for e in events if e["type"] == "tool_call_arguments")
    assert "<tool_call" not in content
    assert starts and starts[0]["name"] == "get_weather"
    assert json.loads(args) == {"city": "Tokyo"}


def test_incremental_metadata_and_arguments_arrive_before_close_and_initial_reasoning():
    parser = IncrementalAssistantParser(
        tools=TOOLS, initial_reasoning=True, id_factory=lambda i: f"req_{i}")
    first = parser.feed("already thought</think><tool_call><function=get_weather><parameter=city>Tok")
    assert any(e["type"] == "reasoning_content" and e["delta"] == "already thought"
               for e in first)
    assert any(e["type"] == "tool_call_start" and e["id"] == "req_0" for e in first)
    assert any(e["type"] == "tool_call_arguments" for e in first)
    second = parser.feed("yo</parameter></function></tool_call>")
    assert not any("<tool_call" in e.get("delta", "") for e in first + second)
    deltas = [event_to_chat_delta(e) for e in first + second]
    tool_deltas = [d for d in deltas if "tool_calls" in d]
    assert tool_deltas and tool_deltas[0]["tool_calls"][0]["function"]["name"] == "get_weather"


@pytest.mark.parametrize("thought", ["thought", "\nthought\n", " \tthought \t",
                                     "思考 α😺\n", "", "\n \t"])
@pytest.mark.parametrize("chunking", ["whole", "character", "byte"])
def test_initial_thinking_is_streamed_once_with_padding_and_utf8(thought, chunking):
    raw = thought + "</think>answer"
    if chunking == "whole":
        chunks = [raw]
    elif chunking == "character":
        chunks = list(raw)
    else:
        encoded = raw.encode("utf-8")
        chunks = [encoded[i:i + 1] for i in range(len(encoded))]
    parser = IncrementalAssistantParser(initial_reasoning=True)
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    result = parser.finish()
    events.extend(result["events"])
    reasoning = "".join(e["delta"] for e in events if e["type"] == "reasoning_content")
    assert reasoning == thought
    assert result["reasoning_content"] == (thought.strip() or None)
    assert result["content"] == "answer"


def test_padded_initial_thinking_does_not_repeat_at_split_close_or_finish():
    parser = IncrementalAssistantParser(initial_reasoning=True)
    events = parser.feed("\nthought\n")
    # Thought text arrives before the closing tag, not only at finish.
    assert "".join(e["delta"] for e in events if e["type"] == "reasoning_content") == "\nthought\n"
    for chunk in ["</thi", "nk", ">", "answer", ""]:
        later = parser.feed(chunk)
        assert not any(e["type"] == "reasoning_content" for e in later)
        events.extend(later)
    result = parser.finish()
    assert not any(e["type"] == "reasoning_content" for e in result["events"])
    assert result["reasoning_content"] == "thought"
    assert result["content"] == "answer"


@pytest.mark.parametrize("initial", ["\nfirst\n", "\n \t"])
def test_reconciling_initial_thinking_preserves_later_thinking_and_tools(initial):
    parser = IncrementalAssistantParser(initial_reasoning=True, tools=TOOLS)
    events = parser.feed(initial + "</think>")
    # Later genuine thought blocks must survive suppression of the first one.
    later = []
    for chunk in ["<think>second</thi", "nk>",
                  '<tool_call><function=get_weather><parameter=city>"Tokyo"',
                  "</parameter></function></tool_call>"]:
        later.extend(parser.feed(chunk))
    result = parser.finish()
    later.extend(result["events"])
    streamed_initial = "".join(e["delta"] for e in events if e["type"] == "reasoning_content")
    assert streamed_initial == initial
    suffix = "".join(e["delta"] for e in later if e["type"] == "reasoning_content")
    assert suffix == ("\nsecond" if initial.strip() else "second")
    assert result["reasoning_content"] == ("first\nsecond" if initial.strip() else "second")
    events.extend(later)
    arguments = "".join(e["delta"] for e in events if e["type"] == "tool_call_arguments")
    assert json.loads(arguments) == {"city": "Tokyo"}
    assert len([e for e in events if e["type"] == "tool_call_start"]) == 1


def test_partial_second_tool_call_never_leaks_raw_xml():
    parser = IncrementalAssistantParser(tools=TOOLS)
    events = parser.feed(
        "answer<tool_call><function=get_weather><parameter=city>A</parameter>"
        "</function></tool_call><tool_call><function=get_weather><parameter=city>B")
    text = "".join(e.get("delta", "") for e in events if e["type"] == "content")
    assert "<tool_call" not in text
    assert any(e["type"] == "tool_call_start" and e["index"] == 0 for e in events)
    assert any(e["type"] == "tool_call_start" and e["index"] == 1 for e in events)


def test_stream_xml_boundary_newline_and_literal_think_markup_in_string():
    parser = IncrementalAssistantParser(tools=TOOLS)
    chunks = [
        '<tool_call>\n<function=get_weather>\n<parameter=city>\n',
        '"Tokyo"\n</parameter>\n</function>\n</tool_call>',
    ]
    events = sum((parser.feed(chunk) for chunk in chunks), [])
    result = parser.finish()
    args = "".join(e["delta"] for e in events + result["events"]
                    if e["type"] == "tool_call_arguments")
    assert json.loads(args) == {"city": "Tokyo"}
    parsed = parse_assistant_output(
        '<tool_call><function=get_weather><parameter=city>"<think>x</think>"'
        '</parameter></function></tool_call>')
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"])["city"] == "<think>x</think>"


def test_template_quoted_angle_string_and_required_tool_choice():
    tools = [{"type": "function", "function": {
        "name": "f", "parameters": {"type": "object", "required": ["s"],
        "properties": {"s": {"type": "string"}}}}}]
    parsed = parse_assistant_output(
        '<tool_call><function=f><parameter=s>"a < b"</parameter></function></tool_call>',
        tools=tools)
    templ = messages_for_template([{"role": "assistant", "content": None,
                                    "tool_calls": parsed["tool_calls"]}], tools)
    assert templ[0]["tool_calls"][0]["function"]["arguments"] == {"s": "a < b"}


def test_literal_parameter_closing_tag_survives_full_and_incremental_parse():
    raw = '<tool_call><function=get_weather><parameter=city>"line1\\n東京\\n</parameter>"'
    raw += '</parameter></function></tool_call>'
    parsed = parse_assistant_output(raw)
    assert json.loads(parsed["tool_calls"][0]["function"]["arguments"]) == {
        "city": "line1\n東京\n</parameter>"}
    parser = IncrementalAssistantParser()
    events = []
    for pos in range(0, len(raw), 4):
        events.extend(parser.feed(raw[pos:pos + 4]))
    final = parser.finish()
    args = "".join(e["delta"] for e in events + final["events"]
                    if e["type"] == "tool_call_arguments")
    assert json.loads(args) == {"city": "line1\n東京\n</parameter>"}


def test_stream_scanner_ignores_xml_openers_inside_nested_and_nullable_values():
    tools = [{"type": "function", "function": {
        "name": "complex", "parameters": {"type": "object",
        "properties": {
            "payload": {"type": "object", "properties": {
                "text": {"type": "string"},
                "items": {"type": "array", "items": {"type": "string"}},
            }},
            "maybe": {"type": ["string", "null"]},
            "choice": {"anyOf": [{"type": "string"}, {"type": "number"}]},
        }, "required": ["payload", "maybe", "choice"]}}}]
    raw = (
        '<tool_call><function=complex>\n'
        '<parameter=payload>{"text":"literal <tool_call><function=decoy>'
        '</function></tool_call> </parameter>","items":['
        '"<parameter=decoy>","<function=decoy>"]}</parameter>\n'
        '<parameter=maybe>null</parameter>\n'
        '<parameter=choice>7</parameter></function></tool_call>'
    )
    full = parse_assistant_output(raw, tools=tools)
    expected = json.loads(full["tool_calls"][0]["function"]["arguments"])
    parser = IncrementalAssistantParser(tools=tools)
    events = []
    for pos in range(0, len(raw), 7):
        events.extend(parser.feed(raw[pos:pos + 7]))
    events.extend(parser.finish()["events"])
    streamed = "".join(e["delta"] for e in events
                         if e["type"] == "tool_call_arguments")
    assert json.loads(streamed) == expected
    assert len([e for e in events if e["type"] == "tool_call_start"]) == 1


def test_incomplete_quoted_xml_is_buffered_and_rejected_only_at_finish():
    parser = IncrementalAssistantParser()
    events = parser.feed('<tool_call><function=get_weather><parameter=city>"line </para')
    assert not any("<tool_call" in e.get("delta", "") for e in events)
    with pytest.raises(ProtocolError):
        parser.finish()


def test_sse_arguments_match_full_parse_with_newline_before_quote():
    raw = ('<tool_call>\n<function=get_weather>\n<parameter=city>\n'
           '"SPEC.md"\n</parameter>\n</function>\n</tool_call>')
    full = parse_assistant_output(raw, tools=TOOLS)
    parser = IncrementalAssistantParser(tools=TOOLS)
    events = []
    # Boundary chunks intentionally isolate the XML newline from the opening
    # quote and split the escaped argument stream into tiny fragments.
    for chunk in ('<tool_call>\n<function=get_weather>\n<parameter=city>\n',
                  '"SPE', 'C.md"\n</parameter>\n</function>\n</tool_call>'):
        events.extend(parser.feed(chunk))
    events.extend(parser.finish()["events"])
    streamed = "".join(e["delta"] for e in events
                         if e["type"] == "tool_call_arguments")
    assert json.loads(streamed) == json.loads(
        full["tool_calls"][0]["function"]["arguments"])


def test_schema_casting_preserves_declared_strings_and_required_arguments():
    tools = [{"type": "function", "function": {
        "name": "f", "parameters": {"type": "object", "required": ["s", "n"],
        "properties": {"s": {"type": "string"}, "n": {"type": "integer"}}}}}]
    parsed = parse_assistant_output(
        "<tool_call><function=f><parameter=s>123</parameter><parameter=n>7</parameter>"
        "</function></tool_call>", tools=tools)
    templ = messages_for_template([{"role": "assistant", "content": None,
                                    "tool_calls": parsed["tool_calls"]}], tools)
    args = templ[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {"s": "123", "n": 7}
    with pytest.raises(ProtocolError):
        parse_assistant_output("<tool_call><function=f><parameter=s>x</parameter></function></tool_call>",
                               tools=tools)


def test_nonstream_response_has_structured_tool_calls_and_finish_reason():
    parsed = parse_assistant_output('<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>')
    response = build_chat_response(parsed, response_id="x", model="m", created=1)
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}

