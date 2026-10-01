"""Structured-output and tool-call constraints for the EXL3 HTTP server.

The server uses Qwen's XML tool-call dialect.  This module deliberately keeps
the transport and generation jobs out of the implementation: it validates an
OpenAI request, builds an llguidance grammar, and returns a small plan that the
HTTP layer can attach to a generation job.  The grammar is active at the
``<tool_call>`` special token, so ordinary reasoning remains unconstrained for
``tool_choice=auto``.  Each parameter is emitted as a JSON value inside the
Qwen ``<parameter=...>`` element; this makes the JSON Schema constraint apply
to the value itself instead of relying on a post-generation check.

Imports of the model tokenizer, torch and llguidance are lazy.  The validation
helpers are consequently safe to use from CPU-only request tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

try:
    from .protocol import ProtocolError, normalize_tool_definitions, validate_tool_choice
except ImportError:  # pragma: no cover - useful when loaded as a standalone file
    class ProtocolError(ValueError):
        pass
    normalize_tool_definitions = None
    validate_tool_choice = None


_PARAMETER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
_SUPPORTED_RESPONSE_FORMATS = {"json_object", "json_schema"}
_MAX_PARALLEL_TOOL_CALLS = 8

# These keywords constrain relationships between multiple object members.
# Qwen's XML dialect emits each member as an independent parameter element, so
# such relationships cannot be represented safely by the per-parameter
# fragments below.  Reject them instead of claiming that the LLGuidance
# grammar enforces a schema it cannot express.
_UNSUPPORTED_TOOL_OBJECT_KEYS = {
    "$ref", "$dynamicRef", "allOf", "anyOf", "oneOf", "not",
    "if", "then", "else", "dependentRequired", "dependentSchemas",
    "dependencies", "minProperties", "maxProperties", "propertyNames",
    "patternProperties", "unevaluatedProperties", "unevaluatedItems",
}


def _error(message: str) -> ProtocolError:
    return ProtocolError(message)


def _schema_copy(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise _error(f"{label} must be a JSON Schema object")
    # jsonschema gives better diagnostics for malformed schemas.  It is an
    # optional dependency in the server image, so retain a useful structural
    # fallback when it is absent.
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(value)
    except ImportError:
        if "type" in value and not isinstance(value["type"], (str, list)):
            raise _error(f"{label}.type must be a string or array")
    except Exception as exc:
        raise _error(f"invalid {label}: {exc}") from exc
    return json.loads(json.dumps(value, ensure_ascii=False))


def normalize_response_format(response_format: Any) -> dict | None:
    """Return ``{"type": ..., "schema": ...}`` for OpenAI response_format.

    ``json_object`` is intentionally represented as ``type: object``.  An
    unknown format is an HTTP 400-worthy error rather than an ignored hint.
    """
    if response_format is None:
        return None
    if not isinstance(response_format, dict):
        raise _error("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return None
    if kind not in _SUPPORTED_RESPONSE_FORMATS:
        raise _error("response_format.type must be text, json_object or json_schema")
    if kind == "json_object":
        # OpenAI's json_object mode promises a JSON object, with no user
        # properties specified.  LLGuidance can enforce this directly.
        return {"type": kind, "schema": {"type": "object"}}
    # Also accept the normalized representation returned by this function.
    # This keeps validation helpers composable for the HTTP layer, which often
    # stores the plan's normalized schema alongside the parsed response.
    if "schema" in response_format and "json_schema" not in response_format:
        return {
            "type": kind,
            "name": response_format.get("name"),
            "strict": bool(response_format.get("strict", False)),
            "schema": _schema_copy(response_format["schema"], "response schema"),
        }
    definition = response_format.get("json_schema")
    if not isinstance(definition, dict):
        raise _error("response_format.json_schema must be an object")
    schema = definition.get("schema")
    if schema is None:
        raise _error("response_format.json_schema.schema is required")
    return {
        "type": kind,
        "name": definition.get("name"),
        "strict": bool(definition.get("strict", False)),
        "schema": _schema_copy(schema, "response_format.json_schema.schema"),
    }


def _schema_validator(schema: dict, label: str):
    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # pragma: no cover - dependency is present in release image
        return None
    try:
        return Draft202012Validator(schema)
    except Exception as exc:
        raise _error(f"invalid {label}: {exc}") from exc


def validate_response_format_output(text: str, response_format: Any) -> Any:
    """Parse and validate a generated response-format payload.

    This is a safety net for transport/parser errors.  The generation filter
    produced by :func:`prepare_constraints` remains the primary guarantee.
    """
    normalized = normalize_response_format(response_format)
    if normalized is None:
        return None
    if not isinstance(text, str):
        raise _error("structured output must be text")
    try:
        value = json.loads(text)
    except (TypeError, ValueError) as exc:
        raise _error(f"generated response is not valid JSON: {exc}") from exc
    validator = _schema_validator(normalized["schema"], "response schema")
    if validator is not None:
        errors = sorted(validator.iter_errors(value), key=lambda e: list(e.path))
        if errors:
            path = ".".join(str(p) for p in errors[0].path) or "$"
            raise _error(f"generated response violates response schema at {path}: {errors[0].message}")
    elif normalized["schema"].get("type") == "object" and not isinstance(value, dict):
        raise _error("generated response must be a JSON object")
    return value


def _tool_map(tools: Iterable[dict] | None) -> dict[str, dict]:
    if normalize_tool_definitions is None:
        raise _error("tool protocol helpers are unavailable")
    normalized = normalize_tool_definitions(list(tools or []))
    result = {}
    for tool in normalized:
        fn = tool["function"]
        params = _schema_copy(fn.get("parameters", {"type": "object", "properties": {}}),
                              f"tool {fn['name']} parameters")
        unsupported = sorted(_UNSUPPORTED_TOOL_OBJECT_KEYS.intersection(params))
        if "propertyNames" in unsupported and _unconstrained_property_names(params["propertyNames"]):
            unsupported.remove("propertyNames")
        if unsupported:
            names = ", ".join(unsupported)
            raise _error(
                f"tool {fn['name']} parameters use unsupported object-level schema keyword(s): {names}; "
                "the XML parameter grammar cannot guarantee those cross-field constraints"
            )
        if params.get("type", "object") != "object":
            raise _error(f"tool {fn['name']} parameters must have type object")
        properties = params.get("properties", {})
        if not isinstance(properties, dict):
            raise _error(f"tool {fn['name']} parameters.properties must be an object")
        for name in properties:
            if not isinstance(name, str) or not _PARAMETER_NAME.fullmatch(name):
                raise _error(f"tool {fn['name']} has an invalid parameter name: {name!r}")
        required = params.get("required", [])
        if not isinstance(required, list) or any(x not in properties for x in required):
            raise _error(f"tool {fn['name']} parameters.required must name properties")
        result[fn["name"]] = {"definition": tool, "schema": params}
    return result


def validate_tool_arguments(calls: Iterable[dict], tools: Iterable[dict] | None) -> list[dict]:
    """Parse and validate OpenAI tool call arguments against tool schemas."""
    tool_map = _tool_map(tools)
    parsed = []
    for index, call in enumerate(calls or []):
        if not isinstance(call, dict):
            raise _error(f"tool call {index} must be an object")
        fn = call.get("function", call)
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise _error(f"tool call {index} has no function name")
        name = fn["name"]
        if name not in tool_map:
            raise _error(f"tool call references unknown function: {name}")
        raw = fn.get("arguments", {})
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError) as exc:
            raise _error(f"tool {name} arguments are not valid JSON: {exc}") from exc
        validator = _schema_validator(tool_map[name]["schema"], f"tool {name} schema")
        if validator is not None:
            errors = sorted(validator.iter_errors(args), key=lambda e: list(e.path))
            if errors:
                path = ".".join(str(p) for p in errors[0].path) or "$"
                raise _error(f"tool {name} arguments violates schema at {path}: {errors[0].message}")
        elif not isinstance(args, dict):
            raise _error(f"tool {name} arguments must be a JSON object")
        parsed.append({"name": name, "arguments": args, "id": call.get("id")})
    return parsed


def _literal(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def _unconstrained_property_names(schema: Any) -> bool:
    # Every JSON object key is already a string. Other name constraints must
    # remain intact (and fail closed if the compiler cannot implement them).
    return schema is True or schema == {} or schema == {"type": "string"}


def _normalize_compiler_schema(node: Any) -> None:
    """Remove only redundant property-name checks at actual schema positions."""
    if not isinstance(node, dict):
        return
    if "propertyNames" in node and _unconstrained_property_names(node["propertyNames"]):
        del node["propertyNames"]
    for key in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        children = node.get(key)
        if isinstance(children, dict):
            for child in children.values():
                _normalize_compiler_schema(child)
    for key in ("items", "additionalProperties", "additionalItems", "unevaluatedProperties",
                "unevaluatedItems", "propertyNames", "not", "contains", "if", "then", "else"):
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                _normalize_compiler_schema(child)
        else:
            _normalize_compiler_schema(children)
    for key in ("allOf", "anyOf", "oneOf", "prefixItems"):
        children = node.get(key)
        if isinstance(children, list):
            for child in children:
                _normalize_compiler_schema(child)
    # Do not walk const/enum/default/examples: they contain literal user data.


def _llguidance_schema(schema: dict) -> dict:
    """Copy a schema and disable unbounded JSON whitespace in LLGuidance.

    Tool schemas and response schemas used for post-validation remain
    untouched. The compiler-only extension prevents a malformed model from
    spending the whole response emitting whitespace before a primitive value.
    """
    compiled = json.loads(json.dumps(schema, ensure_ascii=False))
    _normalize_compiler_schema(compiled)
    guidance = compiled.get("x-guidance", {})
    if not isinstance(guidance, dict):
        guidance = {}
    guidance["whitespace_flexible"] = False
    guidance["lenient"] = False
    guidance["coerce_one_of"] = False
    guidance["item_separator"] = ","
    guidance["key_separator"] = ":"
    guidance.pop("whitespace_pattern", None)
    compiled["x-guidance"] = guidance
    return compiled


def _json_fragment(schema: dict) -> str:
    # %json is a real llguidance Lark extension.  It delegates the fragment to
    # the JSON Schema compiler, so nested arrays, enums and object schemas are
    # constrained by the same engine as response_format.
    return "%json " + json.dumps(
        _llguidance_schema(schema), ensure_ascii=False, separators=(",", ":")
    )


def _parameter_fragment(parent: dict, name: str) -> dict:
    """Make a standalone JSON Schema for one XML parameter value.

    A property such as ``{"$ref": "#/$defs/City"}`` is compiled separately
    from its parent object.  Carrying the parent's local definitions into the
    fragment preserves that reference and also keeps a root ``$id`` available
    for relative references.  Child definitions override inherited names.
    """
    value = parent.get("properties", {}).get(name)
    if not isinstance(value, dict):
        raise _error(f"tool parameter {name} schema must be an object")
    value = json.loads(json.dumps(value, ensure_ascii=False))
    for key in ("$defs", "definitions"):
        inherited = parent.get(key)
        if inherited is None:
            continue
        if not isinstance(inherited, dict):
            raise _error(f"tool parameters.{key} must be an object")
        local = value.get(key, {})
        if not isinstance(local, dict):
            raise _error(f"tool parameter {name}.{key} must be an object")
        value[key] = {**inherited, **local}
    for key in ("$id", "$schema"):
        if key in parent and key not in value:
            value[key] = parent[key]
    # Validate the assembled fragment now so an unresolved local reference is
    # reported as a request error before a generation job is started.
    _schema_copy(value, f"tool parameter {name}")
    return value


def _tool_body_expression(schema: dict) -> str:
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    chunks = []
    for name in properties:
        parameter = " ".join([
            _literal(f"<parameter={name}>\n"),
            _json_fragment(_parameter_fragment(schema, name)),
            _literal("\n</parameter>\n"),
        ])
        # Independent optional blocks keep grammar size linear and permit
        # omission even for tools such as Firecrawl with dozens of settings.
        chunks.append(parameter if name in required else f"( {parameter} )?")
    chunks.extend([_literal("</function>\n"), "</tool_call>"])
    return " ".join(chunks)


def _tool_body_grammar(schema: dict) -> str:
    """Return a standalone body grammar for diagnostics and tests."""
    return "start: " + _tool_body_expression(schema)


def _tool_grammar(
    tool_map: dict[str, dict], names: list[str], *,
    include_marker: bool = False, parallel: bool = False,
    max_calls: int = _MAX_PARALLEL_TOOL_CALLS,
) -> str:
    """Build one-call or bounded contiguous multi-call XML grammar.

    The first marker is consumed by the trigger/prefix path.  For parallel
    calls, the grammar then accepts up to ``max_calls - 1`` additional
    ``<tool_call>`` blocks.  Optional tails keep the matcher accepting after
    every complete call, so EOS is valid after one through eight calls while a
    ninth marker remains masked.
    """
    if not 1 <= max_calls <= _MAX_PARALLEL_TOOL_CALLS:
        raise _error(f"max_parallel_tool_calls must be between 1 and {_MAX_PARALLEL_TOOL_CALLS}")
    rules = []
    for index, name in enumerate(names):
        body = _tool_body_expression(tool_map[name]["schema"])
        rules.append(f"{_literal(f'<function={name}>\n')} ( {body} )")
    choice = "(" + " | ".join(rules) + ")"
    start = ('<tool_call> "\\n" ' if include_marker else "") + choice
    if parallel and max_calls > 1:
        # Expand the bounded repetition instead of using *: llguidance can
        # then mask the ninth marker while still accepting EOS after any
        # completed call.
        tail = " ( <tool_call> " + choice + " )?"
        start += tail * (max_calls - 1)
    # Keep call bodies in the root grammar. A side grammar ending in the
    # user-defined </tool_call> token cannot reliably hand its following
    # newline back to the outer grammar in llguidance.
    return json.dumps({"grammars": [{
        "name": "tool_start", "lark_grammar": "start: " + start,
    }]}, ensure_ascii=False, separators=(",", ":"))


def _tool_call_grammar(tool_map: dict[str, dict], names: list[str]) -> str:
    """Grammar for one call, restarted by the parallel wrapper per trigger."""
    rules = []
    for name in names:
        body = _tool_body_expression(tool_map[name]["schema"])
        rules.append(f"{_literal(f'<function={name}>\n')} ( {body} )")
    choice = "(" + " | ".join(rules) + ")"
    # The marker token itself is consumed by the wrapper. Qwen emits a newline
    # before the function element; required/named calls may already have that
    # newline in the generation prefix, so it remains optional here.
    grammar = 'start: ( "\\n" | /[ \\t]+/ )? ' + choice
    return json.dumps({"grammars": [{"name": "tool_start", "lark_grammar": grammar}]},
                      ensure_ascii=False, separators=(",", ":"))


def _eos_token_ids(tokenizer: Any) -> list[int]:
    config = getattr(tokenizer, "config", None)
    values = getattr(config, "eos_token_id_list", None) or []
    if not values:
        value = getattr(tokenizer, "eos_token_id", None)
        values = [] if value is None else [value]
    return [int(value) for value in values if value is not None]


@dataclass
class ConstraintPlan:
    """Request-scoped generation constraint and parser contract."""

    kind: str = "none"
    schema: dict | None = None
    tools: list[dict] = field(default_factory=list)
    tool_choice: Any = "none"
    parallel_tool_calls: bool = True
    max_parallel_tool_calls: int = _MAX_PARALLEL_TOOL_CALLS
    thinking: bool = True
    trigger_token: int | None = None
    filter_spec: dict | None = None
    generation_prefix: str | None = None
    generation_suffix: str | None = None
    template_instructions: str = ""
    requires_tool_call: bool = False
    filters: list[Any] = field(default_factory=list)

    def build_filters(self, tokenizer) -> list[Any]:
        """Instantiate the native LLGuidanceFilter for this plan."""
        if not self.filter_spec:
            self.filters = []
            return self.filters
        if self.filter_spec.get("filter_type") == "parallel_tool_calls":
            try:
                from .structured_filter import ParallelToolCallFilter
                self.filters = [ParallelToolCallFilter(tokenizer, **{
                    key: value for key, value in self.filter_spec.items()
                    if key not in {"filter_type", "llg_grammar"}
                })]
            except (ImportError, ModuleNotFoundError) as exc:
                raise _error("structured generation requires the llguidance package") from exc
            except Exception as exc:
                raise _error(f"failed to compile structured-output grammar: {exc}") from exc
            return self.filters
        try:
            from exllamav3.generator.filter import LLGuidanceFilter
        except (ImportError, ModuleNotFoundError) as exc:
            raise _error("structured generation requires the llguidance package") from exc
        kwargs = dict(self.filter_spec)
        kwargs["tokenizer"] = tokenizer
        try:
            self.filters = [LLGuidanceFilter(**kwargs)]
        except Exception as exc:
            raise _error(f"failed to compile structured-output grammar: {exc}") from exc
        return self.filters


def _token_id(tokenizer: Any, text: str, label: str) -> int:
    if tokenizer is None or not hasattr(tokenizer, "single_id"):
        raise _error(f"{label} constraint requires a tokenizer with single_id()")
    token_id = tokenizer.single_id(text)
    if token_id is None:
        raise _error(f"tokenizer has no single token for {text!r}; cannot safely activate {label} constraint")
    return int(token_id)


def prepare_constraints(
    tokenizer,
    tools: Iterable[dict] | None = None,
    tool_choice: Any = None,
    parallel_tool_calls: bool = True,
    response_format: Any = None,
    *,
    thinking: bool = True,
    max_parallel_tool_calls: int = _MAX_PARALLEL_TOOL_CALLS,
    compile_filters: bool = True,
) -> ConstraintPlan:
    """Build the generation plan for OpenAI structured output and tools.

    Tool constraints and response-format constraints are intentionally mutually
    exclusive in one assistant turn.  A tool call has an argument schema, while
    a normal answer has a response schema; silently applying one to the other
    would produce an invalid protocol.  The HTTP layer should return this
    ``ProtocolError`` as status 400.
    """
    normalized_response = normalize_response_format(response_format)
    if not 1 <= max_parallel_tool_calls <= _MAX_PARALLEL_TOOL_CALLS:
        raise _error(f"max_parallel_tool_calls must be between 1 and {_MAX_PARALLEL_TOOL_CALLS}")
    tool_map = _tool_map(tools)
    normalized_tools = [entry["definition"] for entry in tool_map.values()]
    choice = validate_tool_choice(tool_choice, normalized_tools,
                                  parallel_tool_calls=parallel_tool_calls) if normalized_tools else (
                                      "none" if tool_choice in (None, "none") else
                                      (_ for _ in ()).throw(_error("tool_choice requires tools")))
    if normalized_response is not None and normalized_tools and choice != "none":
        raise _error("response_format cannot be combined with tool_choice other than none")

    if normalized_response is not None:
        trigger = _token_id(tokenizer, "</think>", "response_format") if thinking else None
        plan = ConstraintPlan(
            kind="response_format", schema=normalized_response["schema"],
            tools=normalized_tools, tool_choice="none", parallel_tool_calls=parallel_tool_calls,
            max_parallel_tool_calls=max_parallel_tool_calls,
            thinking=thinking, trigger_token=trigger,
            filter_spec={"trigger_token": trigger, "eos_after_completed": True,
                         "json_schema": _llguidance_schema(normalized_response["schema"])},
            template_instructions="Emit only a JSON value satisfying the requested response schema after reasoning.",
        )
        if compile_filters:
            plan.build_filters(tokenizer)
        return plan

    if not normalized_tools or choice == "none":
        return ConstraintPlan(kind="none", tools=normalized_tools, tool_choice="none",
                              parallel_tool_calls=parallel_tool_calls, thinking=thinking,
                              max_parallel_tool_calls=max_parallel_tool_calls)

    selected = [choice["function"]["name"]] if isinstance(choice, dict) else list(tool_map)
    trigger = _token_id(tokenizer, "<tool_call>", "tool calling")
    is_required = choice == "required" or isinstance(choice, dict)
    # For required/named calls the HTTP layer may prepend this exact marker to
    # the generation prompt.  Auto mode waits for the trigger token, preserving
    # unconstrained reasoning and normal prose until a tool call is selected.
    # Chat templates with thinking enabled leave the generation cursor inside
    # <think>.  Close that region in the prompt before forcing a required call;
    # the filter itself consumes only the subsequent <tool_call> marker.
    required_prefix = (("</think>\n\n" if thinking else "") + "<tool_call>\n") if is_required else None
    if parallel_tool_calls:
        call_grammar = _tool_call_grammar(tool_map, selected)
        filter_spec = {
            "filter_type": "parallel_tool_calls",
            "trigger_token": trigger,
            "call_grammar": call_grammar,
            "llg_grammar": call_grammar,
            "eos_token_ids": _eos_token_ids(tokenizer),
            "max_calls": max_parallel_tool_calls,
            "required_first": is_required,
        }
    else:
        filter_spec = {
            "trigger_token": None if is_required else trigger,
            "prefix_str": "<tool_call>\n" if is_required else None,
            "consume_prefix": is_required,
            "eos_after_completed": True,
            "llg_grammar": _tool_grammar(
                tool_map, selected, include_marker=is_required,
                parallel=False, max_calls=max_parallel_tool_calls,
            ),
        }
    plan = ConstraintPlan(
        kind="tools", tools=normalized_tools, tool_choice=choice,
        parallel_tool_calls=parallel_tool_calls, thinking=thinking,
        max_parallel_tool_calls=max_parallel_tool_calls,
        trigger_token=trigger, filter_spec=filter_spec,
        generation_prefix=required_prefix, requires_tool_call=is_required,
        template_instructions=(
            "When calling a function, emit Qwen XML <tool_call> blocks. "
            "For multiple requested calls, emit all blocks consecutively. "
            "Encode string parameters as JSON strings including their quotes; "
            "each parameter value must satisfy its schema, with no text after a call."
        ),
    )
    if compile_filters:
        plan.build_filters(tokenizer)
    return plan


__all__ = [
    "ConstraintPlan",
    "normalize_response_format",
    "prepare_constraints",
    "validate_response_format_output",
    "validate_tool_arguments",
]
