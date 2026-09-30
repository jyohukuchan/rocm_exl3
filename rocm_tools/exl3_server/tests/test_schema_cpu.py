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
    class _Config:
        eos_token_id_list = [248044, 248046]

    class _Backend:
        def __init__(self, tokenizer_json: str = ""):
            self._tokenizer_json = tokenizer_json

        def to_str(self):
            return self._tokenizer_json

    def single_id(self, text: str) -> int:
        return {"<tool_call>": 248058, "</think>": 248069}[text]

    eos_token_id = 248044
    config = _Config()
    tokenizer = _Backend()


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
    assert plan.filter_spec["json_schema"]["type"] == "object"
    assert plan.filter_spec["json_schema"]["x-guidance"]["whitespace_flexible"] is False
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
    assert spec["filter_type"] == "parallel_tool_calls"
    assert spec["max_calls"] == 8
    assert plan.generation_prefix is None
    grammar = spec["call_grammar"]
    assert "get_weather" in grammar
    assert "%json" in grammar
    assert "</tool_call>" in grammar
    assert "integer" in grammar
    assert "whitespace_flexible" in grammar


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
        parallel_tool_calls=False,
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


def test_llguidance_rejects_unbounded_whitespace_before_primitive_value():
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
    ll_tokenizer = LLTokenizer(tokenizer_path.read_text())
    matcher = LLMatcher(ll_tokenizer, grammar_from("llguidance", plan.filter_spec["llg_grammar"]))
    malformed = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\n"
        "\n \r\n</parameter>\n</function>\n</tool_call>"
    )
    assert any(not matcher.consume_token(token)
               for token in ll_tokenizer.tokenize_str(malformed))


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
    grammar = grammar_from("llguidance", plan.filter_spec["call_grammar"])
    call = (
        "<function=get_weather>\n<parameter=city>\n\"Tokyo\"\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    # The wrapper restarts this one-call matcher after every trigger. The
    # newline is therefore outside the matcher while waiting, exactly as it
    # is in the native Qwen template.
    for index in range(8):
        matcher = LLMatcher(ll_tokenizer, grammar)
        block = call if index == 0 else "\n" + call
        for token in ll_tokenizer.tokenize_str(block):
            assert matcher.consume_token(token), matcher.get_error()
        assert matcher.is_stopped()
    assert plan.filter_spec["max_calls"] == 8

    bad = LLMatcher(ll_tokenizer, grammar)
    malformed = "\n<function=get_weather>\n<parameter=city>\n\n"
    assert any(not bad.consume_token(token)
               for token in ll_tokenizer.tokenize_str(malformed))


def test_parallel_wrapper_tracks_real_tokens_newlines_cap_and_rewind():
    """Exercise the always-active wrapper with a real Qwen LLTokenizer."""
    try:
        import numpy as np
        import torch
        from llguidance import LLMatcher, LLTokenizer, grammar_from
        from rocm_tools.exl3_server.structured_filter import ParallelToolCallFilter
    except ImportError:
        pytest.skip("llguidance/torch is installed only in the ROCm server image")
    tokenizer_path = Path(os.environ.get(
        "EXL3_SCHEMA_TEST_TOKENIZER",
        "/home/homelab1/datapool/rocm-exl3-rdna2/models/qwen38-flash-next-exl3-3.05bpw/tokenizer.json",
    ))
    if not tokenizer_path.exists():
        pytest.skip("Qwen tokenizer fixture is not available")
    tokenizer_json = tokenizer_path.read_text()
    fake_tokenizer = FakeTokenizer()
    fake_tokenizer.tokenizer = FakeTokenizer._Backend(tokenizer_json)
    ll_tokenizer = LLTokenizer(tokenizer_json)
    plan = prepare_constraints(fake_tokenizer, tools=TOOLS, tool_choice="auto",
                               parallel_tool_calls=True, compile_filters=False)
    call_grammar = plan.filter_spec["call_grammar"]

    class MatcherFilter:
        def __init__(self, tokenizer, *, llg_grammar, **_kwargs):
            self.tokenizer = LLTokenizer(tokenizer.tokenizer.to_str())
            self.grammar = grammar_from("llguidance", llg_grammar)
            self.matcher = LLMatcher(self.tokenizer, self.grammar)
            self.is_active = True

        def attach(self, job):
            self.job = job

        def reset(self):
            self.matcher = LLMatcher(self.tokenizer, self.grammar)
            self.is_active = True

        def feed(self, token):
            assert self.matcher.consume_token(int(token)), self.matcher.get_error()

        def is_completed(self):
            return self.matcher.is_stopped()

        def get_next_logit_mask(self):
            words = (self.tokenizer.vocab_size + 31) // 32
            data = np.empty(words, dtype=np.int32)
            self.matcher.unsafe_compute_mask_ptr(data.ctypes.data, data.nbytes)
            return torch.from_numpy(data).unsqueeze(0)

    class DummyGenerator:
        padded_vocab_size = ll_tokenizer.vocab_size

    class DummyJob:
        generator = DummyGenerator()

    def make_filter():
        f = ParallelToolCallFilter(
            fake_tokenizer, call_grammar=call_grammar,
            trigger_token=248058, eos_token_ids=[248044, 248046], max_calls=8,
            inner_factory=MatcherFilter,
        )
        f.attach(DummyJob())
        f.reset()
        return f

    call = (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\n\"Tokyo\"\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    wrapper = make_filter()

    def allowed(token):
        mask = wrapper.get_next_logit_mask().view(-1)
        word = int(mask[int(token) >> 5].item()) & 0xFFFFFFFF
        return bool(word & (1 << (int(token) & 31)))

    for index in range(8):
        block = call if index == 0 else "\n" + call
        for token in ll_tokenizer.tokenize_str(block):
            assert allowed(token), (index, token)
            wrapper.feed(token)
    assert wrapper._completed_calls == 8
    assert allowed(248044)
    assert not allowed(248058)
    assert wrapper.feed(248044) is True

    replay = make_filter()
    first_ids = ll_tokenizer.tokenize_str(call)
    second_ids = ll_tokenizer.tokenize_str("\n" + call)
    for token in first_ids + second_ids:
        replay.feed(token)
    assert replay._completed_calls == 2
    replay.rewind(len(second_ids))
    assert replay._completed_calls == 1
    for token in second_ids:
        replay.feed(token)
    assert replay._completed_calls == 2

    with pytest.raises(ValueError, match="EOS"):
        ParallelToolCallFilter(
            fake_tokenizer, call_grammar=call_grammar, trigger_token=248058,
            eos_token_ids=[], inner_factory=MatcherFilter,
        )
