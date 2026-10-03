# OpenCode placeholder tool arguments — 2026-10-03

The local OrcaRouter Qwen3.8 Flash Next EXL3 pack returned placeholder strings
instead of usable shell commands and paths. Generation and HTTP requests
completed successfully; shell execution subsequently failed with exit 127 or
file tools reported nonexistent paths. Restarting OpenCode alone would not
address this failure.

## Evidence

The affected OpenCode session was `ses_eff463898ffeofOScdkQiP6JF5`.
In the API audit for `api-20261003-155916-dfdd`, 19 of the first 42 tool calls
already contained `__code_mode_executable_quotewquotquot` in their arguments.
This rules out replacement solely in OpenCode's tool execution layer.

A fresh OpenCode session successfully executed an explicitly requested `pwd`.
Capturing its actual request showed 13 ordinary tool definitions and no
placeholder in the system prompt or schemas. Replaying that request with the
original natural-language request to inspect OpenCode's logs reproduced a
different placeholder, `$$NDLE_DEFAULT_FIRST_TOKEN$$`, in `shell.command`.
Consequently, the existing conversation history was not required to reproduce
the problem.

| Probe | Observed result before the change |
| --- | --- |
| OpenCode request, explicitly ask for `pwd` | Real shell execution, exit 0 |
| Captured tools + original log-inspection request | Placeholder shell argument |
| Same tool/instruction prompt, 10,202 tokens, through unconstrained text completion | Native Qwen XML with a real `ls` command |
| Single-call grammar (`parallel_tool_calls=false`) | Another unusable argument, `command` |
| Add concrete JSON-string examples; original request | Real `ls` command |
| Same examples; independently ask to read the README | `read` with `path=README.md` |

The constrained path requires JSON-quoted strings inside Qwen XML parameters.
The unconstrained model preferred native unquoted XML strings even with the
existing quoted example in the template. Restricting the format could therefore
produce a schema-valid string with unusable contents. JSON Schema cannot tell
whether an otherwise valid string is the intended command.

The old and new local packs have identical tokenizer and chat-template files.
Historical calls using the previous pack did not show these placeholders. This
investigation does **not** isolate the contributions of source weights,
quantization, or MTP; it establishes the failing constrained-generation path
and a prompt-level mitigation.

## Change

The server's tool instructions now explain that values must be actual commands,
paths, or other arguments, and illustrate JSON quoting with `pwd` and
`README.md`. The examples describe parameter serialization; they do not add
tools. Grammar enforcement, post-generation schema validation, and tool
execution behavior remain intact.

This is an empirically checked mitigation, not a guarantee that a model can
never invent a valid-looking but incorrect argument. Raw request captures and
conversation logs remain local rather than being published with this report.

## Validation after deployment

Restarted the API as `api-20261003-171710-8764`, preserving the OrcaRouter pack,
V620 TP2, K5/V4, 786,432-token shared cache, batch 3, fixed one-token MTP, and
xhigh reasoning setting.

- Replayed the previously failing captured request without any extra
  client-side hints: returned a real `ls` command.
- OpenCode session `ses_eff272717ffe7GTc6dXI2gydyC` completed a read-only request
  with two successful calls: `read` of the README and `shell` listing the log
  directory (exit 0), followed by a normal final answer.
- Existing schema, protocol, and client-fixture CPU tests: 52 passed,
  10 skipped (optional dependencies unavailable on the host).

OpenCode's background service and session database did not require a restart
or modification. The inference service is running with the updated instructions.
