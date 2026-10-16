"""Check a running Hy-Memory HTTP server without executing generated tools."""

import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3983")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    def request(path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(args.url + path, data=payload,
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=180).read().decode()

    for _ in range(120):
        try:
            health = json.loads(request("/health"))
            break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError("Server did not become ready")
    props = json.loads(request("/props"))
    defaults = props["default_template_kwargs"]
    assert defaults["reasoning_effort"] == "no_think" and defaults["enable_thinking"] is False
    templates = []
    for effort, native, suffix in [("no_think", "no_think", "<think></think>"),
                                   ("high", "high", "<think>"), ("xhigh", "high", "<think>")]:
        rendered = json.loads(request("/apply-template", {
            "messages": [{"role": "user", "content": "ping"}],
            "chat_template_kwargs": {"reasoning_effort": effort},
        }))["prompt"]
        assert f"reasoning_effort:{native}" in rendered and rendered.endswith(suffix)
        assert ":opensource" not in rendered
        templates.append({"requested": effort, "native": native, "prompt": rendered})
    schema = {"type": "object", "properties": {"memories": {"type": "array", "items": {"type": "string"}}},
              "required": ["memories"], "additionalProperties": False}
    base = {"model": "hy-memory", "temperature": 0, "max_tokens": 192,
            "messages": [{"role": "system", "content": "Extract explicit persistent user facts only."},
                         {"role": "user", "content": "My name is Nora and I live in Lisbon."}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "memory", "strict": True, "schema": schema}}}
    samples = []
    for stream in (False, True):
        start = time.monotonic()
        raw = request("/v1/chat/completions", {**base, "stream": stream})
        if stream:
            chunks = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith("data: ") and line[6:] != "[DONE]"]
            text = "".join(c["choices"][0].get("delta", {}).get("content", "") for c in chunks if c.get("choices"))
            assert not any(c["choices"][0].get("delta", {}).get("reasoning_content") for c in chunks if c.get("choices"))
        else:
            data = json.loads(raw)
            text = data["choices"][0]["message"]["content"]
            assert not data["choices"][0]["message"].get("reasoning_content")
        memories = json.loads(text)["memories"]
        assert "nora" in " ".join(memories).lower() and "lisbon" in " ".join(memories).lower()
        samples.append({"task": "structured_memories", "stream": stream, "response": text,
                        "passed": True, "wall_s": time.monotonic() - start})

    tools = [{"type": "function", "function": {"name": "save_memory", "parameters": {
        "type": "object", "properties": {"fact": {"type": "string"}, "revision": {"type": "integer", "const": 2}},
        "required": ["fact", "revision"], "additionalProperties": False}}}]
    for stream in (False, True):
        raw = request("/v1/chat/completions", {
            "model": "hy-memory", "temperature": 0, "max_tokens": 192, "stream": stream,
            "messages": [{"role": "user", "content": "Call save_memory to save the fact that Nora lives in Lisbon. Use revision 2."}],
            "tools": tools, "tool_choice": {"type": "function", "function": {"name": "save_memory"}},
            "parallel_tool_calls": False, "reasoning_effort": "no_think",
        })
        if stream:
            chunks = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith("data: ") and line[6:] != "[DONE]"]
            calls = [call for c in chunks if c.get("choices")
                     for call in c["choices"][0].get("delta", {}).get("tool_calls", [])]
            arguments = "".join(c["function"].get("arguments", "") for c in calls)
            name = next(c["function"]["name"] for c in calls if c["function"].get("name"))
            assert any(c["choices"][0].get("finish_reason") == "tool_calls" for c in chunks if c.get("choices"))
        else:
            data = json.loads(raw)
            assert data["choices"][0]["finish_reason"] == "tool_calls"
            fn = data["choices"][0]["message"]["tool_calls"][0]["function"]
            name, arguments = fn["name"], fn["arguments"]
        value = json.loads(arguments)
        assert name == "save_memory" and value["revision"] == 2 and "lisbon" in value["fact"].lower()
        samples.append({"task": "native_tool_call", "stream": stream, "function": name,
                        "arguments": value, "passed": True, "tool_executed": False})
    result = {"passed": True, "health": health, "defaults": defaults, "templates": templates,
              "samples": samples, "runtime": props.get("runtime")}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"passed": True, "checks": len(samples), "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
