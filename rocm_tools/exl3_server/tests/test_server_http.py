#!/usr/bin/env python3
"""HTTP contract tests with fake tokenizer/jobs; no model or GPU inference."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
try:
    from rocm_tools.exl3_server import server
except Exception as exc:
    pytest.skip(f"server import unavailable: {exc}", allow_module_level=True)

import torch

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "weather",
    "parameters": {"type": "object", "required": ["city"],
                   "properties": {"city": {"type": "string"}}}}}]


class FakeTokenizer:
    bos_token_id = 1

    def __init__(self):
        self.calls = []

    def hf_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return torch.tensor([[1, 2, 3]], dtype=torch.long)

    def encode(self, value, **kwargs):
        return torch.tensor([[1] * max(1, len(str(value)))], dtype=torch.long)


class FakeJob:
    def __init__(self, events):
        self.events = list(events)
        self.cancelled = False
        self.new_tokens = 0

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for event in self.events:
            self.new_tokens = event.get("new_tokens", self.new_tokens)
            yield event

    async def cancel(self):
        self.cancelled = True


@pytest.fixture
def http_state(monkeypatch):
    tok = FakeTokenizer()
    args = SimpleNamespace(api_key=None, host="127.0.0.1", port=0,
                           temperature=0.7, min_p=0.0, top_k=0, top_p=1.0,
                           repetition_penalty=1.0, presence_penalty=0.0,
                           frequency_penalty=0.0, penalty_range=0,
                           temperature_first=False, adaptive_target=1.0,
                           adaptive_decay=0.0, xtc_probability=0.0,
                           xtc_threshold=0.1, dry_multiplier=0.0,
                           dry_base=1.75, dry_allowed_length=2,
                           dry_penalty_last_n=0, loop_window=0)
    names = ("args", "tokenizer", "model_name", "context_length",
             "has_chat_template", "default_template_kwargs", "generator",
             "runtime", "max_output_tokens")
    old = {name: getattr(server.state, name) for name in names}
    server.state.args = args
    server.state.tokenizer = tok
    server.state.model_name = "fake-model"
    server.state.context_length = 4096
    server.state.max_output_tokens = 32
    server.state.has_chat_template = True
    server.state.default_template_kwargs = {}
    server.state.runtime = None
    events = [{"text": "hello", "eos": True, "eos_reason": "stop",
               "new_tokens": 1, "prompt_tokens": 3, "cached_tokens": 0,
               "time_prefill": 0.1, "time_generate": 0.1}]
    jobs = []

    def make_job(*_args, **_kwargs):
        job = FakeJob(events)
        jobs.append(job)
        return job

    monkeypatch.setattr(server, "make_job", make_job)
    monkeypatch.setattr(server.structured, "prepare_constraints",
                        lambda *_a, **_kw: SimpleNamespace(
                            filters=[], template_instructions="", generation_prefix=""))
    yield SimpleNamespace(tokenizer=tok, jobs=jobs, events=events)
    for name, value in old.items():
        setattr(server.state, name, value)


async def request_json(method, path, payload):
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, json=payload)


@pytest.mark.asyncio
async def test_ordinary_chat_nonstream_usage_and_limits(http_state):
    response = await request_json("POST", "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] == 4
    server.state.max_output_tokens = 2
    response = await request_json("POST", "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 99,
    })
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_tool_nonstream_and_template_history(http_state):
    http_state.events[:] = [{"text": "</think><tool_call><function=get_weather>"
                             "<parameter=city>Tokyo</parameter></function></tool_call>",
                             "eos": True, "eos_reason": "stop", "new_tokens": 4,
                             "prompt_tokens": 3, "cached_tokens": 0}]
    response = await request_json("POST", "/v1/chat/completions", {
        "model": "fake-model", "tools": TOOLS,
        "messages": [{"role": "user", "content": "weather"}],
    })
    assert response.status_code == 200
    choice = response.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]) == {"city": "Tokyo"}

    response = await request_json("POST", "/v1/chat/completions", {
        "model": "fake-model", "tools": TOOLS,
        "messages": [{"role": "user", "content": "weather"},
                     {"role": "assistant", "content": None, "tool_calls": [
                         {"id": "call_old", "type": "function",
                          "function": {"name": "get_weather", "arguments": "{\"city\":\"Osaka\"}"}}
                     ]},
                     {"role": "tool", "tool_call_id": "call_old", "content": "rain"}],
    })
    assert response.status_code == 200
    history = http_state.tokenizer.calls[-1][0]
    assert isinstance(history[1]["tool_calls"][0]["function"]["arguments"], dict)


@pytest.mark.asyncio
async def test_tool_controls_and_streaming_partial_arguments(http_state):
    http_state.events[:] = [{"text": "</think><tool_call><function=get_weather>"
                             "<parameter=city>Tokyo</parameter></function></tool_call>",
                             "eos": True, "eos_reason": "stop", "new_tokens": 4,
                             "prompt_tokens": 3, "cached_tokens": 0}]
    for choice in ("required", {"type": "function", "function": {"name": "get_weather"}}):
        response = await request_json("POST", "/v1/chat/completions", {
            "model": "fake-model", "tools": TOOLS, "tool_choice": choice,
            "messages": [{"role": "user", "content": "weather"}],
        })
        assert response.status_code == 200
    for bad in ("required", {"type": "function", "function": {"name": "missing"}}):
        response = await request_json("POST", "/v1/chat/completions", {
            "model": "fake-model", "tools": [] if bad == "required" else TOOLS,
            "tool_choice": bad, "messages": [{"role": "user", "content": "x"}],
        })
        assert response.status_code == 400

    http_state.events[:] = [
        {"text": "</think><tool_call><function=get_weather><parameter=city>To", "eos": False},
        {"text": "kyo</parameter></function></tool_call>", "eos": True,
         "eos_reason": "max_new_tokens", "new_tokens": 2,
         "prompt_tokens": 3, "cached_tokens": 0},
    ]
    response = await request_json("POST", "/v1/chat/completions", {
        "model": "fake-model", "tools": TOOLS, "stream": True,
        "messages": [{"role": "user", "content": "weather"}],
    })
    assert response.status_code == 200
    assert '"tool_calls"' in response.text
    assert '"finish_reason": "tool_calls"' in response.text
    assert "[DONE]" in response.text


@pytest.mark.asyncio
async def test_http_errors_context_model_and_cancellation(http_state):
    response = await request_json("POST", "/v1/chat/completions", {
        "model": "unknown", "messages": [{"role": "user", "content": "x"}],
    })
    assert response.status_code == 404
    server.state.context_length = 1
    response = await request_json("POST", "/v1/chat/completions", {
        "model": "fake-model", "messages": [{"role": "user", "content": "x"}],
    })
    assert response.status_code == 400

    class CancelJob(FakeJob):
        async def _iter(self):
            raise asyncio.CancelledError
            yield  # pragma: no cover

    job = CancelJob([])
    with pytest.raises(asyncio.CancelledError):
        await server.collect_job(job)
    assert job.cancelled

