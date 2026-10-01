# Client request fixtures

`librechat_firecrawl.json` preserves the request fields, 27 Firecrawl MCP tool
schemas, and parallel assistant/tool turn structure from the LibreChat request
used to reproduce the 2026-10-01 long-conversation/title transition failure.
Message text, reasoning, tool results, string arguments, and call IDs have been
replaced. Schema descriptions, examples, titles and defaults are omitted.
No headers, credentials, original user questions or retrieved page bodies are
stored here.

`opencode.json` was captured from OpenCode 2.0.12 against a local mock API in an
isolated project/config/data directory. It preserves the 12 standard tool
schemas, provider body and transport options. It did not send a request to the
live model or execute tools. System/user text and schema annotations have been
replaced or removed; authentication headers were never recorded.

These fixtures exercise API adaptation, actual model template rendering and
LLGuidance compilation. HTTP tests use fake generation jobs; they do not prove
GPU kernel correctness. Native transition verification is recorded separately
in `doc/qsa_request_transition_fix.md`.
