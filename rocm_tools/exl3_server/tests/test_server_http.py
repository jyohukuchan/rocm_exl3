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


@pytest.mark.asyncio
async def test_native_metrics_are_available_in_json_and_sse_without_usage_opt_in(http_state):
    http_state.events[:] = [{"text": "hello", "eos": True, "eos_reason": "stop",
                           "new_tokens": 10, "prompt_tokens": 100,
                           "cached_tokens": 80, "time_prefill": .2,
                           "time_generate": .5}]
    body = {"model": "fake-model", "enable_thinking": False,
            "messages": [{"role": "user", "content": "hello"}]}
    plain = await request_json("POST", "/v1/chat/completions", body)
    assert plain.json()["exl3_metrics"]["prefill_tokens_per_second"] == 100
    assert plain.json()["exl3_metrics"]["output_tokens_per_second"] == 20
    stream = await request_json("POST", "/v1/chat/completions", {**body, "stream": True})
    chunks = [json.loads(line[5:].strip()) for line in stream.text.splitlines()
              if line.startswith("data:") and line[5:].strip() != "[DONE]"]
    measured = [c for c in chunks if "exl3_metrics" in c]
    assert len(measured) == 1
    for key, value in plain.json()["exl3_metrics"].items():
        if key == "total_seconds":
            assert measured[0]["exl3_metrics"][key] >= 0
        else:
            assert measured[0]["exl3_metrics"][key] == value
    assert "usage" not in measured[0]


async def request_json(method, path, payload):
    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, json=payload)


def chat_payloads(response, stream):
    if stream:
        chunks = [json.loads(line[6:]) for line in response.text.splitlines()
                  if line.startswith("data: {")]
        text = "".join(c["choices"][0]["delta"].get("content", "")
                       for c in chunks if c.get("choices"))
        final = next(c for c in chunks if "exl3_metrics" in c)
        return text, final
    final = response.json()
    return final["choices"][0]["message"]["content"], final


@pytest.mark.asyncio
@pytest.mark.parametrize("effort,native,thinking", [
    ("none", None, False), ("minimal", "low", True), ("low", "low", True),
    ("medium", "medium", True), ("high", "xhigh", True), ("xhigh", "xhigh", True)])
@pytest.mark.parametrize("nested", [False, True])
async def test_reasoning_aliases_and_default_xhigh(http_state, effort, native, thinking, nested):
    server.state.default_template_kwargs = {"enable_thinking": True, "reasoning_effort": "xhigh"}
    body = {"messages": [{"role": "user", "content": "hi"}], "response_format": {"type": "text"}}
    body["chat_template_kwargs" if nested else "reasoning_effort"] = {"reasoning_effort": effort} if nested else effort
    response = await request_json("POST", "/v1/chat/completions", body)
    assert response.status_code == 200
    kwargs = http_state.tokenizer.calls[-1][1]
    assert kwargs.get("reasoning_effort") == native
    assert kwargs["enable_thinking"] is thinking
    response = await request_json("POST", "/v1/chat/completions", {"messages": body["messages"]})
    assert response.status_code == 200
    assert http_state.tokenizer.calls[-1][1]["reasoning_effort"] == "xhigh"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_explicit_text_response_allows_tools_and_timing_footer(http_state, stream):
    response = await request_json("POST", "/v1/chat/completions", {
        "messages": [{"role": "developer", "content": "follow rules"}, {"role": "user", "content": "hi"}],
        "response_format": {"type": "text"}, "include_timings": True, "stream": stream,
        "tools": TOOLS})
    assert response.status_code == 200
    text, _ = chat_payloads(response, stream)
    assert "exl3-timings:v2" in text
    assert http_state.tokenizer.calls[-1][0][0] == {"role": "system", "content": "follow rules"}


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["image_url", "input_audio", "file", "video", "unexpected"])
async def test_unsupported_content_is_rejected_instead_of_silently_lost(http_state, kind):
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "describe"}, {"type": kind}]}]}
    for path in ("/v1/chat/completions", "/apply-template"):
        response = await request_json("POST", path, body)
        assert response.status_code == 400
        assert "Unsupported content type" in response.json()["error"]["message"]
    assert not http_state.jobs


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cli,option,expected", [
    (False, None, False), (False, True, True),
    (True, None, True), (True, False, False)])
async def test_timing_footer_opt_in_and_wall_time(http_state, monkeypatch, stream, cli, option, expected):
    server.state.args.include_timings = cli
    clock = [100.]
    monkeypatch.setattr(server.time, "perf_counter", lambda: clock[0])
    original = server.make_job

    def queued_job(*args, **kwargs):
        clock[0] += 3.25  # Wall time includes queue/preparation, not just phase times.
        return original(*args, **kwargs)

    monkeypatch.setattr(server, "make_job", queued_job)
    http_state.events[:] = [{"text": "hello", "eos": True, "eos_reason": "stop",
                           "new_tokens": 10, "prompt_tokens": 100, "cached_tokens": 80,
                           "time_prefill": .2, "time_generate": .5,
                           "accepted_draft_tokens": 3, "rejected_draft_tokens": 1}]
    body = {"messages": [{"role": "user", "content": "hi"}], "stream": stream,
            "stream_options": {"include_usage": True}}
    if option is not None:
        body["include_timings"] = option
    response = await request_json("POST", "/v1/chat/completions", body)
    assert response.status_code == 200
    text, final = chat_payloads(response, stream)
    assert final["exl3_metrics"]["total_seconds"] == 3.25
    if expected:
        assert "Prefill: 100.00 tok/s | Decode: 20.00 tok/s | Total: 3.25 s" in text
        assert text.startswith("hello\n\n")
        assert "Draft: 75.00%" in text
        assert final["usage"]["completion_tokens"] == 10
    else:
        assert text == "hello"


@pytest.mark.asyncio
async def test_timing_footer_removed_before_tokenization_and_apply_template(http_state, monkeypatch):
    body = {"messages": [{"role": "user", "content": "hi"}], "include_timings": True}
    first = await request_json("POST", "/v1/chat/completions", body)
    answer = first.json()["choices"][0]["message"]["content"]
    assert "exl3-timings:v2" in answer
    body.update(messages=body["messages"] + [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        {"role": "user", "content": "continue"}], include_timings=False)
    second = await request_json("POST", "/v1/chat/completions", body)
    assert second.status_code == 200
    assert http_state.tokenizer.calls[-1][0][1]["content"] == "hello"
    monkeypatch.setattr(http_state.tokenizer, "hf_render_chat_template",
                        lambda messages, **kwargs: json.dumps(messages), raising=False)
    rendered = await request_json("POST", "/apply-template", {"messages": body["messages"]})
    assert json.loads(rendered.json()["prompt"])[1]["content"] == "hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("kind", ["tool", "json"])
async def test_timing_footer_preserves_tool_and_json_contract(http_state, stream, kind):
    body = {"messages": [{"role": "user", "content": "hi"}],
            "include_timings": True, "stream": stream}
    if kind == "tool":
        text = ('checking<tool_call><function=get_weather><parameter=city>"Tokyo"'
                '</parameter></function></tool_call>')
        body["tools"] = TOOLS
    else:
        text = '{"result":"hello"}'
        body["response_format"] = {"type": "json_object"}
    http_state.events[:] = [{"text": text, "eos": True, "eos_reason": "stop",
                           "new_tokens": 10, "prompt_tokens": 3}]
    response = await request_json("POST", "/v1/chat/completions", body)
    assert response.status_code == 200
    answer, final = chat_payloads(response, stream)
    assert "exl3-timings" not in answer
    assert final["choices"][0]["finish_reason"] == ("tool_calls" if kind == "tool" else "stop")
    if kind == "json":
        assert json.loads(answer) == {"result": "hello"}


@pytest.mark.asyncio
@pytest.mark.parametrize("draft_tokens", [0, 1, 4])
async def test_near_full_cache_reserves_speculative_window(http_state, monkeypatch, draft_tokens):
    server.state.generator = SimpleNamespace(generator=SimpleNamespace(num_draft_tokens=draft_tokens))
    captured = []

    def checked_job(req, ids, max_new, *_args, **_kwargs):
        # Model the same page capacity enforced by the real enqueue path.
        reserved = ids.shape[-1] + max_new + 1 + draft_tokens
        assert (reserved + 255) // 256 <= server.state.context_length // 256
        captured.append(max_new)
        return FakeJob(http_state.events)

    monkeypatch.setattr(server, "make_job", checked_job)
    response = await request_json("POST", "/v1/completions", {
        "model": "fake-model", "prompt": "x" * (4094 - draft_tokens),
        "add_bos": False, "max_tokens": 32,
    })
    assert response.status_code == 200
    assert captured == [1]
    response = await request_json("POST", "/v1/completions", {
        "model": "fake-model", "prompt": "x" * (4095 - draft_tokens),
        "add_bos": False, "max_tokens": 1,
    })
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert captured == [1]  # Rejected before allocating/enqueuing another job.


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
async def test_raw_completion_n_choices_aggregates_usage(http_state):
    response = await request_json("POST", "/v1/completions", {
        "model": "fake-model", "prompt": "hi", "n": 2, "max_tokens": 4,
    })
    assert response.status_code == 200
    usage = response.json()["usage"]
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 5


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


def test_qwen_template_adapter_canonicalizes_json_parameter_history(http_state):
    source = (
        "HEAD<tool_call>\n<parameter=example_parameter_1>\\nvalue_1\\n</parameter>\\n"
        "<parameter=example_parameter_2>\\nThis is the value for the second parameter\\n"
        "that can span\\nmultiple lines\\n</parameter>TAIL"
    )
    http_state.tokenizer.tokenizer_config_dict = {"chat_template": source}
    write_tool = {"type": "function", "function": {
        "name": "write", "parameters": {"type": "object", "properties": {
            "text": {"type": "string"}, "count": {"type": "integer"}}}}}
    request = server.ChatCompletionRequest(
        model="fake-model", tools=[write_tool], max_tokens=4,
        messages=[{"role": "user", "content": "write it"},
                  {"role": "assistant", "content": None, "tool_calls": [{
                      "id": "call_keep", "type": "function", "function": {
                          "name": "write",
                          "arguments": json.dumps({"text": "line1\n東京", "count": 7},
                                                    ensure_ascii=False)}}]}])
    server.chat_prompt_ids(request)
    messages, kwargs = http_state.tokenizer.calls[-1]
    args = messages[1]["tool_calls"][0]["function"]["arguments"]
    assert messages[1]["tool_calls"][0]["id"] == "call_keep"
    assert args == {"text": '"line1\\n東京"', "count": 7}
    patched = kwargs["chat_template"]
    assert '<parameter=example_parameter_1>\\n"value_1"\\n' in patched
    assert '"This is the value for the second parameter\\\\nthat can span' in patched
    assert source not in patched


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


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/completion"])
async def test_model_failure_is_not_a_successful_empty_response(http_state, monkeypatch, stream, path):
    class FailedJob(FakeJob):
        async def _iter(self):
            raise RuntimeError("PRIVATE_REQUEST_CONTENT")
            yield

    job = FailedJob([])
    monkeypatch.setattr(server, "make_job", lambda *args, **kwargs: job)
    response = await request_json("POST", path, {
        "messages": [{"role": "user", "content": "hi"}], "prompt": "hi", "stream": stream})
    assert response.status_code == (200 if stream else 503)
    assert "inference_failed" in response.text
    assert "PRIVATE_REQUEST_CONTENT" not in response.text
    assert job.cancelled
    assert '"finish_reason": "stop"' not in response.text


@pytest.mark.asyncio
async def test_failed_enqueue_does_not_leave_async_job_registered():
    class RejectQueue:
        def enqueue(self, _job):
            raise ValueError("too large")
    generator = server.AsyncGenerator.__new__(server.AsyncGenerator)
    generator.generator = RejectQueue()
    generator.jobs = {}
    generator.error = None
    async_job = SimpleNamespace(job=object())
    with pytest.raises(ValueError):
        generator.enqueue(async_job)
    assert generator.jobs == {}


@pytest.mark.asyncio
async def test_librechat_tool_history_title_and_new_conversation_in_one_server(http_state):
    from pathlib import Path
    body = json.loads((Path(__file__).parent / "fixtures/librechat_firecrawl.json").read_text())
    server.state.model_name = body["model"]
    conversation = await request_json("POST", "/v1/chat/completions", body)
    assert conversation.status_code == 200
    assert "[DONE]" in conversation.text
    rendered = http_state.tokenizer.calls[-1][0]
    assert len([m for m in rendered if m["role"] == "tool"]) == 8
    for assistant in (m for m in rendered if m.get("tool_calls")):
        assert len(assistant["tool_calls"]) == 2
        assert all(isinstance(c["function"]["arguments"], dict) for c in assistant["tool_calls"])
    for prompt in ("Generate a synthetic conversation title", "新しい会話です"):
        short = await request_json("POST", "/v1/chat/completions", {
            "model": body["model"], "messages": [{"role": "user", "content": prompt}],
            "stream": False, "reasoning_effort": "xhigh"})
        assert short.status_code == 200
        assert short.json()["choices"][0]["message"]["content"] == "hello"
    assert all(not job.cancelled for job in http_state.jobs)


@pytest.mark.asyncio
async def test_async_generator_close_clears_sync_queue_before_waking_consumers():
    class FakeSyncGenerator:
        def __init__(self):
            self.active_jobs = [object()]
            self.pending_jobs = [object()]
            self.clear_calls = 0

        def clear_queue(self):
            self.clear_calls += 1
            self.active_jobs.clear()
            self.pending_jobs.clear()

    class WaitingJob:
        def __init__(self):
            self.results = []

        def put_result(self, result):
            self.results.append(result)

    sync = FakeSyncGenerator()
    agen = server.AsyncGenerator.__new__(server.AsyncGenerator)
    agen.generator = sync
    agen.jobs = {}
    agen.error = None
    agen.condition = asyncio.Condition()
    waiting = WaitingJob()
    agen.jobs[object()] = waiting

    async def idle():
        await asyncio.Event().wait()

    agen.iteration_task = asyncio.create_task(idle())
    await asyncio.sleep(0)
    await agen.close()

    assert sync.clear_calls == 1
    assert sync.active_jobs == [] and sync.pending_jobs == []
    assert len(waiting.results) == 1
    assert agen.jobs == {}

