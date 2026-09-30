"""CPU-only tests for OpenAI structured-output planning.

The tests deliberately use ``compile_filters=False`` by default: the release
container supplies llguidance, while the repository's ordinary Python test
environment need not install its native extension.  Grammar text and all
request/response validation paths are still exercised here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from rocm_tools.exl3_server.protocol import ProtocolError  # noqa: E402
from rocm_tools.exl3_server.schema import (  # noqa: E402
    normalize_response_format,
    prepare_constraints,
    validate_response_format_output,
    validate_tool_arguments,
)


class FakeTokenizer:
    def single_id(self, text: str) -> int:
        return {"<tool_call>": 248058, "</think>": 248069}[text]


TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "minLength": 1},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            "days": {"type": "integer", "minimum": 1, "maximum": 7},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
}}]

NESTED_TOOLS = [{"type": "function", "function": {
    "name": "lookup_city",
    "parameters": {
        "$id": "https://example.invalid/tool/lookup",
        "type": "object",
        "$defs": {
            "City": {"type": "string", "pattern": "^[A-Z][A-Za-z -]+$"},
        },
        "properties": {"city": {"$ref": "#/$defs/City"}},
        "required": ["city"],
        "additionalProperties": False,
    },
}}]


def test_response_format_normalization_and_validation():
    rf = normalize_response_format({
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True,
                         "schema": {"type": "object", "properties": {
                             "ok": {"type": "boolean"}}, "required": ["ok"]}},
    })
    assert rf["schema"]["type"] == "object"
    assert validate_response_format_output('{"ok":true}', rf) == {"ok": True}
    with pytest.raises(ProtocolError, match="violates"):
        validate_response_format_output('{"ok":"yes"}', rf)
    with pytest.raises(ProtocolError, match="not valid JSON"):
        validate_response_format_output("not-json", {"type": "json_object"})


def test_json_object_mode_builds_a_real_constraint_plan():
    plan = prepare_constraints(
        FakeTokenizer(), response_format={"type": "json_object"},
        thinking=True, compile_filters=False,
    )
    assert plan.kind == "response_format"
    assert plan.trigger_token == 248069
    assert plan.filter_spec["json_schema"] == {"type": "object"}
    assert plan.filter_spec["eos_after_completed"] is True


def test_auto_tool_plan_uses_trigger_and_schema_grammar():
    plan = prepare_constraints(
        FakeTokenizer(), tools=TOOLS, tool_choice="auto",
        parallel_tool_calls=True, compile_filters=False,
    )
    spec = plan.filter_spec
    assert plan.kind == "tools"
    assert plan.trigger_token == 248058
    assert spec["trigger_token"] == 248058
    assert spec["eos_after_completed"] is False
    assert plan.generation_prefix is None
    grammar = spec["llg_grammar"]
    assert "get_weather" in grammar
    assert "%json" in grammar
    assert "</tool_call>" in grammar
    assert "integer" in grammar
    assert grammar.count("<tool_call>") == 7  # bounded tail: one through eight calls


def test_required_and_named_tool_plans_force_the_first_marker():
    required = prepare_constraints(
        FakeTokenizer(), tools=TOOLS, tool_choice="required",
        parallel_tool_calls=False, compile_filters=False,
    )
    assert required.requires_tool_call is True
    assert required.generation_prefix == "</think>\n\n<tool_call>\n"
    assert required.filter_spec["trigger_token"] is None
    assert required.filter_spec["consume_prefix"] is True
    assert required.filter_spec["eos_after_completed"] is True

    named = prepare_constraints(
        FakeTokenizer(), tools=TOOLS,
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        compile_filters=False,
    )
    assert named.generation_prefix == "</think>\n\n<tool_call>\n"
    assert "get_weather" in named.filter_spec["llg_grammar"]
    assert "<tool_call>" in required.filter_spec["llg_grammar"]


def test_response_format_and_tools_are_explicitly_rejected_together():
    with pytest.raises(ProtocolError, match="cannot be combined"):
        prepare_constraints(
            FakeTokenizer(), tools=TOOLS, tool_choice="auto",
            response_format={"type": "json_object"}, compile_filters=False,
        )


def test_tool_argument_validation_is_schema_backed():
    good = [{"id": "c0", "function": {"name": "get_weather",
            "arguments": json.dumps({"city": "Tokyo", "units": "celsius", "days": 2})}}]
    parsed = validate_tool_arguments(good, TOOLS)
    assert parsed[0]["arguments"]["days"] == 2
    bad = [{"id": "c1", "function": {"name": "get_weather",
           "arguments": json.dumps({"city": "Tokyo", "days": 99})}}]
    with pytest.raises(ProtocolError, match="violates"):
        validate_tool_arguments(bad, TOOLS)


def test_unsupported_response_format_is_not_silently_ignored():
    with pytest.raises(ProtocolError, match="json_object or json_schema"):
        normalize_response_format({"type": "text"})


def test_nested_defs_are_carried_into_each_parameter_grammar_and_validator():
    plan = prepare_constraints(FakeTokenizer(), tools=NESTED_TOOLS,
                               tool_choice="required", parallel_tool_calls=False,
                               compile_filters=False)
    assert "City" in plan.filter_spec["llg_grammar"]
    good = [{"function": {"name": "lookup_city",
             "arguments": json.dumps({"city": "Tokyo"})}}]
    assert validate_tool_arguments(good, NESTED_TOOLS)[0]["arguments"] == {"city": "Tokyo"}
    bad = [{"function": {"name": "lookup_city",
            "arguments": json.dumps({"city": "tokyo"})}}]
    with pytest.raises(ProtocolError, match="violates"):
        validate_tool_arguments(bad, NESTED_TOOLS)


def test_llguidance_compiles_tool_grammar_when_extension_is_available():
    try:
        from llguidance import grammar_from
    except ImportError:
        pytest.skip("llguidance is installed only in the ROCm server image")
    plan = prepare_constraints(FakeTokenizer(), tools=TOOLS, compile_filters=False)
    compiled = grammar_from("llguidance", plan.filter_spec["llg_grammar"])
    assert compiled


def test_llguidance_matcher_accepts_required_prefix_and_valid_xml_body():
    """Exercise the special-token prefix path with the real Qwen tokenizer."""
    try:
        from llguidance import LLMatcher, LLTokenizer, grammar_from
    except ImportError:
        pytest.skip("llguidance is installed only in the ROCm server image")
    tokenizer_path = Path(os.environ.get(
        "EXL3_SCHEMA_TEST_TOKENIZER",
        "/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-flash-next-exl3-3.05bpw/tokenizer.json",
    ))
    if not tokenizer_path.exists():
        pytest.skip("Qwen tokenizer fixture is not available")
    plan = prepare_constraints(FakeTokenizer(), tools=TOOLS, tool_choice="required",
                               parallel_tool_calls=False, compile_filters=False)
    grammar = grammar_from("llguidance", plan.filter_spec["llg_grammar"])
    ll_tokenizer = LLTokenizer(tokenizer_path.read_text())
    matcher = LLMatcher(ll_tokenizer, grammar)
    output = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=city>\n\"Tokyo\"\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    for token in ll_tokenizer.tokenize_str(output):
        assert matcher.consume_token(token), matcher.get_error()
    assert matcher.is_stopped()
    assert not matcher.is_error(), matcher.get_error()


def test_nested_defs_are_enforced_by_the_real_llguidance_matcher():
    try:
        from llguidance import LLMatcher, LLTokenizer, grammar_from
    except ImportError:
        pytest.skip("llguidance is installed only in the ROCm server image")
    tokenizer_path = Path(os.environ.get(
        "EXL3_SCHEMA_TEST_TOKENIZER",
        "/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-flash-next-exl3-3.05bpw/tokenizer.json",
    ))
    if not tokenizer_path.exists():
        pytest.skip("Qwen tokenizer fixture is not available")
    plan = prepare_constraints(FakeTokenizer(), tools=NESTED_TOOLS, tool_choice="required",
                               parallel_tool_calls=False, compile_filters=False)
    ll_tokenizer = LLTokenizer(tokenizer_path.read_text())
    grammar = grammar_from("llguidance", plan.filter_spec["llg_grammar"])
    matcher = LLMatcher(ll_tokenizer, grammar)
    output = (
        "<tool_call>\n<function=lookup_city>\n<parameter=city>\n"
        "\"Tokyo\"\n</parameter>\n</function>\n</tool_call>"
    )
    for token in ll_tokenizer.tokenize_str(output):
        assert matcher.consume_token(token), matcher.get_error()
    assert matcher.is_stopped()
    assert not matcher.is_error(), matcher.get_error()


def test_cross_field_object_keywords_are_rejected_before_generation():
    unsupported = [{"type": "function", "function": {
        "name": "choose",
        "parameters": {
            "type": "object",
            "oneOf": [{"required": ["a"]}, {"required": ["b"]}],
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        },
    }}]
    with pytest.raises(ProtocolError, match="unsupported object-level"):
        prepare_constraints(FakeTokenizer(), tools=unsupported, tool_choice="auto",
                            compile_filters=False)


def test_parallel_tool_grammar_keeps_all_calls_constrained_and_caps_at_eight():
    try:
        from llguidance import LLMatcher, LLTokenizer, grammar_from
    except ImportError:
        pytest.skip("llguidance is installed only in the ROCm server image")
    tokenizer_path = Path(os.environ.get(
        "EXL3_SCHEMA_TEST_TOKENIZER",
        "/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-flash-next-exl3-3.05bpw/tokenizer.json",
    ))
    if not tokenizer_path.exists():
        pytest.skip("Qwen tokenizer fixture is not available")
    plan = prepare_constraints(FakeTokenizer(), tools=TOOLS, tool_choice="auto",
                               parallel_tool_calls=True, compile_filters=False)
    ll_tokenizer = LLTokenizer(tokenizer_path.read_text())
    matcher = LLMatcher(ll_tokenizer, grammar_from("llguidance", plan.filter_spec["llg_grammar"]))
    call = (
        "<function=get_weather>\n<parameter=city>\n\"Tokyo\"\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    for index in range(8):
        block = call if index == 0 else "<tool_call>" + call
        for token in ll_tokenizer.tokenize_str(block):
            assert matcher.consume_token(token), matcher.get_error()
        if index < 7:
            assert not matcher.is_stopped()
    assert matcher.is_stopped()
    assert not matcher.is_error(), matcher.get_error()
    assert not matcher.consume_token(248058)
