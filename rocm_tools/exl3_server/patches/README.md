# LLGuidance: Firecrawl MCP regex compilation

Validated on 2026-10-01 with the 36 tool definitions exposed by LibreChat v0.8.8:
Firecrawl MCP 3.24.0 (27 tools), Hugging Face (4), and OpenAI Docs (5).

LLGuidance 1.8.0 and stock 1.9.1 reject Firecrawl's domain-name pattern combined
with `maxLength: 253`: `Unable to determine if regex is empty`. This comes from
a hard-coded 10,000-step emptiness proof budget in the JSON Schema compiler;
`LLParserLimits.initial_lexer_fuel` does not change it.

`llguidance-mcp-regex.patch` raises that bounded proof budget to 100,000 and labels
the wheel `1.9.1+exl3.1`. It changes no schema constraints, generation masks,
parser step limits, or GPU kernels. The regex still rejects invalid domains and
strings beyond the original length bound. The EXL3 schema adapter separately
normalizes redundant `propertyNames` checks; both fixes are needed for Firecrawl.

Reproduce against this exact upstream revision:

```bash
git clone https://github.com/guidance-ai/llguidance.git llguidance-mcp
cd llguidance-mcp
git checkout e7c69004694064a01db30ddc7a29a103a5d2f3c7
git apply /ABSOLUTE/rocm_exl3/rocm_tools/exl3_server/patches/llguidance-mcp-regex.patch
python3 -m venv .venv
.venv/bin/pip install maturin==1.15.0
CARGO_BUILD_JOBS=4 .venv/bin/maturin build --release --compatibility linux \
  --interpreter /usr/bin/python3 --out wheels
```

The build requires Rust (the checkout specifies its toolchain). This command
produces a Linux wheel for the machine's architecture and glibc; build on the
inference host/container rather than assuming portability across distributions.
Install it into the inference environment, or an isolated directory placed
first on the server's `PYTHONPATH`, then restart the server. Keep the upstream
[MIT license](https://github.com/guidance-ai/llguidance/blob/main/LICENSE).

Verification used the original schemas and native Qwen tokenizer: all 36 tools
compile with default runtime limits, valid domain strings remain accepted, and
invalid/overlong domain strings remain masked. The EXL3 API suite passes 96 tests.
