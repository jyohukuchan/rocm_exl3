"""Pure OpenAI chat protocol helpers for the exl3 server.

The HTTP layer owns authentication, scheduling and SSE transport.  This module
only normalizes tool history, validates tool request controls, and converts the
Qwen XML assistant format into structured OpenAI tool calls.  It intentionally
never executes a tool.
"""

from __future__ import annotations

import codecs
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


_FUNCTION_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_THINK_RE = re.compile(r"<think\s*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)
_QWEN_CALL_RE = re.compile(
    r"<tool_call\s*>\s*<function=([^>\s]+)\s*>(.*?)</function\s*>\s*</tool_call\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)</parameter\s*>",
    re.IGNORECASE | re.DOTALL,
)
_JSON_CALL_RE = re.compile(
    r"<tool_call\s*>(\s*\{.*?\}\s*)</tool_call\s*>",
    re.IGNORECASE | re.DOTALL,
)
_FUNCTION_START_RE = re.compile(
    r"<tool_call\s*>\s*<function=([^>\s]+)\s*>", re.IGNORECASE | re.DOTALL)
_FUNCTION_ONLY_RE = re.compile(r"<function=([^>\s]+)\s*>", re.IGNORECASE)
_PARAM_STREAM_RE = re.compile(
    r"<parameter=([^>\s]+)\s*>(.*?)(?=</parameter\s*>|<parameter=|</function|</tool_call|$)",
    re.IGNORECASE | re.DOTALL)
_PARAM_START_RE = re.compile(r"<parameter=([^>\s]+)\s*>", re.IGNORECASE)


class ProtocolError(ValueError):
    """A request or generated assistant payload violates the chat contract."""


def _json_arguments(value: Any) -> str:
    if isinstance(value, str):
        # Validate JSON when possible, but preserve a model's raw argument text
        # when it is still incomplete or malformed; the caller can surface it.
        try:
            json.loads(value)
            return value
        except (TypeError, ValueError):
            return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False,
                      separators=(",", ":"))


def _escape_json_fragment(value: str) -> str:
    """Escape a partial string without double-escaping model JSON escapes."""
    return (value.replace('"', '\\"').replace("\b", "\\b")
            .replace("\f", "\\f").replace("\n", "\\n")
            .replace("\r", "\\r").replace("\t", "\\t"))


def _tool_schemas(tools: Iterable[dict] | None) -> dict[str, dict]:
    return {t["function"]["name"]: t["function"].get("parameters", {})
            for t in normalize_tool_definitions(list(tools or []))}


def _coerce_schema(value: Any, schema: dict, *, path: str = "arguments") -> Any:
    """Small JSON-schema coercer used for the Qwen template's typed items."""
    if not isinstance(schema, dict):
        return value
    typ = schema.get("type")
    if typ == "object":
        if not isinstance(value, dict):
            raise ProtocolError(f"{path} must be an object")
        props = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ProtocolError(f"{path}.{key} is required")
        return {k: _coerce_schema(v, props.get(k, {}), path=f"{path}.{k}")
                for k, v in value.items()}
    if typ == "array":
        if not isinstance(value, list):
            raise ProtocolError(f"{path} must be an array")
        return [_coerce_schema(v, schema.get("items", {}), path=f"{path}[]") for v in value]
    if typ == "string":
        return value if isinstance(value, str) else str(value)
    if typ == "integer":
        if isinstance(value, bool):
            raise ProtocolError(f"{path} must be an integer")
        try:
            return int(value)
        except (TypeError, ValueError):
            raise ProtocolError(f"{path} must be an integer") from None
    if typ == "number":
        if isinstance(value, bool):
            raise ProtocolError(f"{path} must be a number")
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ProtocolError(f"{path} must be a number") from None
    if typ == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ProtocolError(f"{path} must be a boolean")
    return value


def _text_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict) and part.get("type") in ("text", "input_text"):
                out.append(str(part.get("text", "")))
        return "".join(out)
    return str(content)


def _remove_top_level_think(text: str) -> tuple[str, str]:
    """Remove reasoning tags before the first tool block only."""
    lower = text.lower()
    marker = -1
    for candidate in [m.start() for m in re.finditer(r"<tool_call\b", lower)]:
        inside = any(m.start() <= candidate < m.end()
                     for m in _THINK_RE.finditer(text))
        if not inside:
            marker = candidate
            break
    prefix = text if marker < 0 else text[:marker]
    suffix = "" if marker < 0 else text[marker:]
    reasoning = "\n".join(m.group(1).strip() for m in _THINK_RE.finditer(prefix)).strip()
    prefix = _THINK_RE.sub("", prefix)
    return reasoning, prefix + suffix


def _find_tag_outside_quotes(text: str, marker: str, start: int = 0, end: int | None = None) -> int:
    """Find an XML marker while ignoring quoted JSON/string contents."""
    end = len(text) if end is None else end
    lower, wanted = text.lower(), marker.lower()
    quote, escaped = False, False
    i = start
    while i < end:
        char = text[i]
        if quote:
            if char == '"' and not escaped:
                quote = False
            escaped = char == "\\" and not escaped
            if char != "\\":
                escaped = False
            i += 1
            continue
        if char == '"':
            quote = True
            i += 1
            continue
        if lower.startswith(wanted, i):
            return i
        i += 1
    return -1


def _parse_qwen_xml_calls(text: str, id_factory: Callable[[int], str] | None = None) -> tuple[list[dict], list[tuple[int, int]]]:
    calls, spans = [], []
    cursor = 0
    while True:
        start = text.lower().find("<tool_call", cursor)
        if start < 0:
            break
        open_end = text.find(">", start)
        if open_end < 0:
            break
        close = _find_tag_outside_quotes(text, "</tool_call", open_end + 1)
        if close < 0:
            break
        block_end = text.find(">", close)
        if block_end < 0:
            break
        fn_match = _FUNCTION_ONLY_RE.search(text, open_end + 1, close)
        if fn_match:
            fn_close = _find_tag_outside_quotes(text, "</function", fn_match.end(), close)
            if fn_close < 0:
                cursor = block_end + 1
                continue
            body = text[fn_match.end():fn_close]
            args = {}
            ppos = 0
            while True:
                rel = _find_tag_outside_quotes(body, "<parameter=", ppos)
                if rel < 0:
                    break
                name_end = body.find(">", rel)
                if name_end < 0:
                    break
                pname = body[rel + len("<parameter="):name_end].strip()
                pclose = _find_tag_outside_quotes(body, "</parameter", name_end + 1)
                if pclose < 0:
                    break
                raw = body[name_end + 1:pclose].strip()
                args[pname] = _decode_parameter_value(raw)
                ptag_end = body.find(">", pclose)
                ppos = len(body) if ptag_end < 0 else ptag_end + 1
            call_id = id_factory(len(calls)) if id_factory else f"call_{len(calls)}"
            calls.append(_tool_call({"id": call_id,
                                     "function": {"name": fn_match.group(1),
                                                   "arguments": args}}, len(calls)))
        else:
            call = _parse_json_call(text[open_end + 1:close], len(calls))
            if call is not None:
                if id_factory:
                    call["id"] = id_factory(len(calls))
                calls.append(call)
        spans.append((start, block_end + 1))
        cursor = block_end + 1
    return calls, spans


def _decode_parameter_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            try:
                return json.loads(raw.replace("\n", "\\n").replace("\r", "\\r"))
            except (TypeError, ValueError):
                return raw[1:-1]
        return raw


def _tool_call(call: Any, index: int = 0) -> dict:
    if not isinstance(call, dict):
        raise ProtocolError("tool_calls entries must be objects")
    fn = call.get("function", call)
    if not isinstance(fn, dict):
        raise ProtocolError("tool_call.function must be an object")
    name = fn.get("name")
    if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
        raise ProtocolError("tool call function name is invalid")
    return {
        "id": str(call.get("id") or f"call_{index}"),
        "type": "function",
        "function": {"name": name, "arguments": _json_arguments(fn.get("arguments", {}))},
    }


def normalize_tool_definitions(tools: Iterable[dict] | None) -> list[dict]:
    """Validate and copy OpenAI function tool definitions."""
    if tools is None:
        return []
    if not isinstance(tools, list):
        raise ProtocolError("tools must be an array")
    out, names = [], set()
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type", "function") != "function":
            raise ProtocolError("only function tools are supported")
        fn = tool.get("function", tool)
        if not isinstance(fn, dict):
            raise ProtocolError("tool.function must be an object")
        name = fn.get("name")
        if not isinstance(name, str) or not _FUNCTION_NAME.fullmatch(name):
            raise ProtocolError("tool function name is invalid")
        if name in names:
            raise ProtocolError(f"duplicate tool function name: {name}")
        names.add(name)
        parameters = fn.get("parameters", {"type": "object", "properties": {}})
        if not isinstance(parameters, dict):
            raise ProtocolError(f"tool {name}: parameters must be an object")
        out.append({"type": "function", "function": {
            "name": name,
            "description": fn.get("description", ""),
            "parameters": parameters,
        }})
    return out


def validate_tool_choice(tool_choice: Any, tools: Iterable[dict] | None = None,
                         *, parallel_tool_calls: bool = True) -> Any:
    """Normalize ``auto``/``none``/``required``/named tool_choice controls."""
    normalized = normalize_tool_definitions(list(tools or []))
    names = {t["function"]["name"] for t in normalized}
    choice = "auto" if tool_choice is None else tool_choice
    if isinstance(choice, str):
        if choice not in {"auto", "none", "required"}:
            raise ProtocolError("tool_choice must be auto, none, required, or a named function")
        if choice == "required" and not names:
            raise ProtocolError("tool_choice=required requires at least one tool")
        return choice
    if not isinstance(choice, dict) or choice.get("type", "function") != "function":
        raise ProtocolError("named tool_choice must be {type:function,function:{name}}")
    fn = choice.get("function")
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
        raise ProtocolError("named tool_choice requires function.name")
    if fn["name"] not in names:
        raise ProtocolError(f"tool_choice names unknown function: {fn['name']}")
    if not parallel_tool_calls:
        return {"type": "function", "function": {"name": fn["name"]}}
    return {"type": "function", "function": {"name": fn["name"]}}


def validate_parallel_tool_calls(tool_calls: Iterable[dict], *, parallel_tool_calls: bool = True) -> None:
    calls = list(tool_calls or [])
    if not parallel_tool_calls and len(calls) > 1:
        raise ProtocolError("multiple tool calls require parallel_tool_calls=true")


def normalize_chat_messages(messages: Iterable[dict]) -> list[dict]:
    """Preserve tool-call IDs, assistant calls, content and reasoning history."""
    if not isinstance(messages, list) or not messages:
        raise ProtocolError("messages must be a non-empty array")
    out = []
    known_call_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            raise ProtocolError("messages entries must be objects")
        role = message.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ProtocolError(f"unsupported message role: {role}")
        item = {"role": role, "content": _text_content(message.get("content"))}
        if "name" in message:
            item["name"] = message["name"]
        if "reasoning_content" in message:
            item["reasoning_content"] = _text_content(message["reasoning_content"])
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ProtocolError("tool messages require tool_call_id")
            if call_id not in known_call_ids:
                raise ProtocolError(f"tool_call_id has no prior assistant tool call: {call_id}")
            item["tool_call_id"] = call_id
        if role == "assistant" and message.get("tool_calls") is not None:
            calls = [_tool_call(c, i) for i, c in enumerate(message["tool_calls"])]
            item["tool_calls"] = calls
            validate_parallel_tool_calls(calls)
            known_call_ids.update(c["id"] for c in calls)
        out.append(item)
    return out


def messages_for_template(messages: Iterable[dict], tools: Iterable[dict] | None = None) -> list[dict]:
    """Return HF-template messages with assistant arguments as typed objects.

    ``normalize_chat_messages`` intentionally keeps OpenAI's JSON argument
    strings. Qwen's actual template iterates ``tool_call.arguments|items``, so
    this separate helper decodes and schema-coerces those strings only at the
    template boundary while retaining IDs, names, reasoning and tool results.
    """
    normalized = normalize_chat_messages(messages)
    schemas = _tool_schemas(tools)
    out = []
    for msg in normalized:
        item = dict(msg)
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            calls = []
            for call in msg["tool_calls"]:
                name = call["function"]["name"]
                if tools is not None and name not in schemas:
                    raise ProtocolError(f"assistant call uses undeclared tool: {name}")
                try:
                    args = json.loads(call["function"]["arguments"])
                except (TypeError, ValueError) as exc:
                    raise ProtocolError(f"tool {name} arguments are not JSON") from exc
                if not isinstance(args, dict):
                    raise ProtocolError(f"tool {name} arguments must be a JSON object")
                args = _coerce_schema(args, schemas.get(name, {}))
                calls.append({"id": call["id"], "type": "function",
                              "function": {"name": name, "arguments": args}})
            item["tool_calls"] = calls
        out.append(item)
    return out


def validate_output_tools(parsed: dict, tools: Iterable[dict] | None = None,
                          *, tool_choice: Any = "auto",
                          parallel_tool_calls: bool = True) -> dict:
    """Validate generated calls against declarations and request choice."""
    declarations = normalize_tool_definitions(list(tools or []))
    schemas = {t["function"]["name"]: t["function"].get("parameters", {})
               for t in declarations}
    calls = list(parsed.get("tool_calls") or [])
    validate_parallel_tool_calls(calls, parallel_tool_calls=parallel_tool_calls)
    for call in calls:
        name = call.get("function", {}).get("name")
        if declarations and name not in schemas:
            raise ProtocolError(f"model called undeclared tool: {name}")
        try:
            args = json.loads(call.get("function", {}).get("arguments", "{}"))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"tool {name} emitted invalid JSON arguments") from exc
        if not isinstance(args, dict):
            raise ProtocolError(f"tool {name} arguments must be an object")
        _coerce_schema(args, schemas.get(name, {}))
    choice = validate_tool_choice(tool_choice, declarations,
                                  parallel_tool_calls=parallel_tool_calls)
    if choice == "required" and not calls:
        raise ProtocolError("tool_choice=required but model emitted no tool call")
    if isinstance(choice, dict):
        wanted = choice["function"]["name"]
        if not any(c.get("function", {}).get("name") == wanted for c in calls):
            raise ProtocolError(f"tool_choice required {wanted}, but model emitted no such call")
    return parsed


def _parse_json_call(raw: str, index: int) -> dict | None:
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    name = value.get("name") or value.get("function", {}).get("name")
    args = value.get("arguments", value.get("function", {}).get("arguments", {}))
    return _tool_call({"id": f"call_{index}", "function": {"name": name, "arguments": args}}, index)


def parse_assistant_output(text: str, *, model_family: str = "qwen38",
                           tools: Iterable[dict] | None = None,
                           tool_choice: Any = "auto",
                           parallel_tool_calls: bool = True,
                           id_factory: Callable[[int], str] | None = None,
                           allow_partial: bool = False) -> dict:
    """Parse Qwen ``<think>`` and function XML without executing tools."""
    if not isinstance(text, str):
        raise ProtocolError("assistant output must be text")
    reasoning, visible = _remove_top_level_think(text)
    calls, spans = _parse_qwen_xml_calls(visible, id_factory=id_factory)
    if not allow_partial and "<tool_call" in visible.lower() and not calls:
        raise ProtocolError("incomplete or malformed tool_call block")
    # Remove parsed blocks while preserving all ordinary assistant text.
    for start, end in sorted(spans, reverse=True):
        visible = visible[:start] + visible[end:]
    content = visible.strip()
    validate_parallel_tool_calls(calls, parallel_tool_calls=parallel_tool_calls)
    result = {
        "content": content or None,
        "reasoning_content": reasoning or None,
        "tool_calls": calls,
        "finish_reason": "tool_calls" if calls else "stop",
    }
    if tools is not None or tool_choice != "auto":
        validate_output_tools(result, tools, tool_choice=tool_choice,
                              parallel_tool_calls=parallel_tool_calls)
    return result


def build_chat_message(parsed: dict) -> dict:
    message = {"role": "assistant", "content": parsed.get("content")}
    if parsed.get("reasoning_content"):
        message["reasoning_content"] = parsed["reasoning_content"]
    if parsed.get("tool_calls"):
        message["tool_calls"] = parsed["tool_calls"]
    return message


def build_chat_response(parsed: dict, *, response_id: str = "chatcmpl-local",
                        model: str = "exl3-model", created: int | None = None,
                        index: int = 0, usage: dict | None = None) -> dict:
    return {"id": response_id, "object": "chat.completion",
            "created": int(time.time()) if created is None else created,
            "model": model,
            "choices": [{"index": index, "message": build_chat_message(parsed),
                         "finish_reason": parsed.get("finish_reason", "stop")}],
            "usage": usage}


def build_chat_chunk(delta: dict, *, response_id: str = "chatcmpl-local",
                     model: str = "exl3-model", created: int | None = None,
                     index: int = 0, finish_reason: str | None = None,
                     usage: dict | None = None) -> dict:
    data = {"id": response_id, "object": "chat.completion.chunk",
            "created": int(time.time()) if created is None else created,
            "model": model,
            "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        data["usage"] = usage
    return data


def event_to_chat_delta(event: dict) -> dict:
    """Convert an IncrementalAssistantParser event to an OpenAI delta."""
    kind = event.get("type")
    if kind == "content":
        return {"content": event.get("delta", "")}
    if kind == "reasoning_content":
        return {"reasoning_content": event.get("delta", "")}
    if kind == "tool_call_start":
        return {"tool_calls": [{
            "index": int(event.get("index", 0)),
            "id": event.get("id"),
            "type": "function",
            "function": {"name": event.get("name", ""), "arguments": ""},
        }]}
    if kind == "tool_call_arguments":
        return {"tool_calls": [{
            "index": int(event.get("index", 0)),
            "function": {"arguments": event.get("delta", "")},
        }]}
    if kind == "tool_call":
        call = event.get("tool_call") or {}
        fn = call.get("function") or {}
        return {"tool_calls": [{
            "index": int(event.get("index", 0)),
            "id": call.get("id"),
            "type": "function",
            "function": {"name": fn.get("name", ""),
                         "arguments": fn.get("arguments", "")},
        }]}
    return {}


@dataclass
class IncrementalAssistantParser:
    """Incremental Qwen parser; buffers incomplete tags and UTF-8 bytes."""
    model_family: str = "qwen38"
    tools: list[dict] | None = None
    initial_reasoning: bool = False
    id_factory: Callable[[int], str] | None = None
    _decoder: Any = field(default_factory=lambda: codecs.getincrementaldecoder("utf-8")())
    _text: str = ""
    _emitted_content: int = 0
    _emitted_reasoning: int = 0
    _emitted_calls: int = 0
    _stream_calls: dict = field(default_factory=dict)
    _initial_reasoning_sent: bool = False
    _initial_reasoning_done: bool = False
    _initial_reasoning_cursor: int = 0
    _initial_reasoning_value: str = ""
    _finished: bool = False

    def feed(self, chunk: str | bytes) -> list[dict]:
        if self._finished:
            raise ProtocolError("cannot feed after finish")
        if isinstance(chunk, bytes):
            chunk = self._decoder.decode(chunk, final=False)
        elif not isinstance(chunk, str):
            raise ProtocolError("stream chunks must be str or UTF-8 bytes")
        self._text += chunk
        initial_events = []
        if self.initial_reasoning and not self._initial_reasoning_done:
            lower_all = self._text.lower()
            close = lower_all.find("</think")
            if close < 0:
                # Hold a partial closing tag, but emit genuine thought text.
                safe_end = len(self._text)
                for i in range(1, 8):
                    if lower_all.endswith("</think"[:i]):
                        safe_end -= i
                        break
                piece = self._text[self._initial_reasoning_cursor:safe_end]
                if piece:
                    initial_events.append({"type": "reasoning_content", "delta": piece})
                    self._initial_reasoning_value += piece
                    self._initial_reasoning_cursor = safe_end
                return initial_events
            piece = self._text[self._initial_reasoning_cursor:close]
            if piece:
                initial_events.append({"type": "reasoning_content", "delta": piece})
                self._initial_reasoning_value += piece
            end_tag = lower_all.find(">", close)
            if end_tag < 0:
                self._initial_reasoning_cursor = close
                return initial_events
            self._initial_reasoning_done = True
            self._initial_reasoning_cursor = end_tag + 1
        parse_input = self._text
        if self.initial_reasoning and self._initial_reasoning_done:
            parse_input = ("<think>" + self._initial_reasoning_value + "</think>"
                           + self._text[self._initial_reasoning_cursor:])
        parsed = parse_assistant_output(parse_input, tools=self.tools,
                                        id_factory=self.id_factory,
                                        allow_partial=True)
        events = initial_events
        events.extend(self._events_from_parsed(parsed))
        events.extend(self._stream_xml_events())
        return events

    def _events_from_parsed(self, parsed: dict) -> list[dict]:
        events = []
        reasoning = parsed.get("reasoning_content") or ""
        if self.initial_reasoning and self._initial_reasoning_value \
                and reasoning.startswith(self._initial_reasoning_value):
            reasoning = reasoning[len(self._initial_reasoning_value):]
        content = parsed.get("content") or ""
        if len(reasoning) > self._emitted_reasoning:
            events.append({"type": "reasoning_content", "delta": reasoning[self._emitted_reasoning:]})
            self._emitted_reasoning = len(reasoning)
        # Hold a trailing partial tool marker/body so tool-looking text is not
        # emitted as ordinary content before the XML block closes.
        safe_content = content
        lower = self._text.lower()
        think_open = lower.rfind("<think") > lower.rfind("</think")
        partial_think = any(lower.endswith("<think"[:i]) for i in range(1, 6))
        partial_think_close = any(lower.endswith("</think"[:i]) for i in range(1, 8))
        partial_tool = any(lower.endswith("<tool_call"[:i]) for i in range(1, 10))
        marker_count = len(re.findall(r"<tool_call\b", lower))
        completed_count = len(list(_QWEN_CALL_RE.finditer(self._text)))
        if think_open or partial_think or partial_think_close:
            safe_content = ""
        elif (marker_count > completed_count) or partial_tool:
            marker = content.lower().find("<tool")
            safe_content = content[:max(0, marker)] if marker >= 0 else ""
        if len(safe_content) > self._emitted_content:
            events.append({"type": "content", "delta": safe_content[self._emitted_content:]})
            self._emitted_content = len(safe_content)
        calls = parsed.get("tool_calls") or []
        xml_count = len(list(_FUNCTION_START_RE.finditer(self._text)))
        for index in range(self._emitted_calls, len(calls)):
            if index < xml_count:
                continue
            events.append({"type": "tool_call", "index": index,
                           "tool_call": calls[index]})
        self._emitted_calls = max(self._emitted_calls, len(calls))
        return events

    def _stream_xml_events(self) -> list[dict]:
        """Emit append-only tool metadata/argument fragments as XML arrives."""
        events = []
        schemas = _tool_schemas(self.tools)
        source = self._text
        if self.initial_reasoning:
            if not self._initial_reasoning_done:
                return []
            source = self._text[self._initial_reasoning_cursor:]
        else:
            # Explicit <think> blocks are excluded from tool scanning.
            if source.lower().rfind("<think") > source.lower().rfind("</think"):
                return []
            _, source = _remove_top_level_think(source)
        starts = list(_FUNCTION_START_RE.finditer(source))
        for index, start in enumerate(starts):
            name = start.group(1)
            state = self._stream_calls.setdefault(index, {
                "id": self.id_factory(index) if self.id_factory else f"call_{index}",
                "name": name, "prefix": False, "params": {}, "closed": False,
                "param_count": 0,
            })
            if not state["prefix"]:
                events.append({"type": "tool_call_start", "index": index,
                               "id": state["id"], "name": name})
                state["prefix"] = True
            next_start = starts[index + 1].start() if index + 1 < len(starts) else len(source)
            body = source[start.end():next_start]
            params = list(_PARAM_START_RE.finditer(body))
            for p in params:
                pname = p.group(1)
                typ = schemas.get(name, {}).get("properties", {}).get(pname, {}).get("type", "string")
                if typ == "string" and body[p.end():].startswith('"'):
                    escaped = False
                    quote = -1
                    for pos, char in enumerate(body[p.end() + 1:], p.end() + 1):
                        if char == '"' and not escaped:
                            quote = pos
                            break
                        escaped = (char == "\\" and not escaped)
                        if char != "\\":
                            escaped = False
                    if quote >= 0:
                        tails = [quote + 1]
                    else:
                        tails = [len(body)]
                else:
                    tails = []
                    for marker in ("</parameter", "<parameter", "</function", "</tool_call"):
                        pos = body.lower().find(marker, p.end())
                        if pos >= 0:
                            tails.append(pos)
                    # A closing tag may itself be split across chunks.
                    lt = body.find("<", p.end())
                    if lt >= 0:
                        tails.append(lt)
                end = min(tails) if tails else len(body)
                raw = body[p.end():end]
                pstate = state["params"].setdefault(
                    pname, {"seen": 0, "started": False, "mode": None})
                complete = body[end:].lower().startswith("</parameter")
                if not pstate["started"]:
                    prefix = ("{" if state["param_count"] == 0 else ",") + json.dumps(pname) + ":"
                    if typ == "string":
                        prefix += '"'
                    events.append({"type": "tool_call_arguments", "index": index,
                                   "delta": prefix})
                    pstate["started"] = True
                    state["param_count"] += 1
                value = raw
                if typ == "string":
                    probe = value.lstrip()
                    if pstate["mode"] is None:
                        if not probe:
                            continue
                        pstate["mode"] = "quoted" if probe.startswith('"') else "plain"
                    if pstate["mode"] == "quoted":
                        normalized = probe[1:] if probe.startswith('"') else probe
                        if complete:
                            normalized = normalized.rstrip()
                        if complete and normalized.endswith('"'):
                            normalized = normalized[:-1]
                        elif not complete and normalized.endswith('"'):
                            # The model's closing JSON quote often arrives
                            # before the XML parameter close; hold it until
                            # completion.
                            normalized = normalized[:-1]
                    else:
                        normalized = probe.rstrip() if complete else probe.rstrip()
                    value_delta = normalized[pstate["seen"]:]
                    if value_delta:
                        events.append({"type": "tool_call_arguments", "index": index,
                                       "delta": _escape_json_fragment(value_delta)})
                    if complete and not pstate.get("closed"):
                        events.append({"type": "tool_call_arguments", "index": index,
                                       "delta": '"'})
                        pstate["closed"] = True
                    pstate["seen"] = len(normalized)
                else:
                    value = value[pstate["seen"]:]
                    if value:
                        events.append({"type": "tool_call_arguments", "index": index,
                                       "delta": value})
                    pstate["seen"] = len(raw)
            closed = (_find_tag_outside_quotes(body, "</tool_call") >= 0
                      and _find_tag_outside_quotes(body, "</function") >= 0)
            if closed and not state["closed"]:
                events.append({"type": "tool_call_arguments", "index": index, "delta": "}"})
                state["closed"] = True
        return events

    def finish(self) -> dict:
        if not self._finished:
            tail = self._decoder.decode(b"", final=True)
            if tail:
                self._text += tail
        parse_input = self._text
        if self.initial_reasoning and self._initial_reasoning_done:
            parse_input = ("<think>" + self._initial_reasoning_value + "</think>"
                           + self._text[self._initial_reasoning_cursor:])
        parsed = parse_assistant_output(parse_input, model_family=self.model_family,
                                        tools=self.tools, id_factory=self.id_factory)
        events = self._events_from_parsed(parsed)
        self._finished = True
        return {"events": events, **parsed}


# Ergonomic aliases for server wiring and downstream callers.
normalize_tool_history = normalize_chat_messages
parse_model_output = parse_assistant_output
AssistantStreamParser = IncrementalAssistantParser
stream_event_to_delta = event_to_chat_delta

