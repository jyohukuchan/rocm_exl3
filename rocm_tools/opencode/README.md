# OpenCode: EXL3 metrics and persistent goals

Tested with OpenCode **2.0.12**, 2026-10-01. The native EXL3 server must include
the `exl3_metrics` response extension. Other servers display no EXL3 statistics.

The footer displays **completed-request averages**:

- PP = uncached prompt tokens / engine prefill seconds.
- TG = generated tokens / engine generation seconds (including thinking,
  tool syntax, and MTP output).
- Cache = reused prompt tokens.

Before the first output it shows `prefill…`; during output it shows
`generating…`. These are client request states, not a live GPU profiler.
Queue/network latency is not used to calculate the displayed rates. Title and
compaction requests are excluded. Missing engine timings are shown as unavailable.
Metrics are kept per session under `$XDG_STATE_HOME/opencode/exl3-metrics` (or
`~/.local/state/opencode/exl3-metrics`), with no prompt/response text or API keys.

## Display plugin

Add this directory as a **directory**, not an individual file, to the project
`opencode.jsonc` server plugin list:

```json
{
  "plugins": [
    {"package": "file:///ABSOLUTE/REPO/rocm_tools/opencode/exl3-status"}
  ]
}
```

Add the same directory to `~/.config/opencode/cli.json` for terminal rendering.
Preserve existing plugin entries and other configuration. Restart the terminal
client after changing its plugin list.

## Goal plugin

The local installation pins `@bybrawe/opencode-goal` **1.3.38** and
`@opencode/plugin` **2.0.12**. The native V2 entry is required; this release's
default and TUI entries are V1 interfaces. The goal package remains separately
MIT-licensed: <https://github.com/ByBrawe/opencode-goal>.

Install the pinned dependencies into a local prefix:

```bash
npm install --prefix ~/.local/share/opencode/rocm-exl3-plugins \
  --ignore-scripts --save-exact \
  @bybrawe/opencode-goal@1.3.38 @opencode/plugin@2.0.12
mkdir -p ~/.local/share/opencode/rocm-exl3-plugins/goal
```

Create `goal/index.ts` containing:

```ts
import plugin from "../node_modules/@bybrawe/opencode-goal/dist/native.js";
import { adaptGoal } from "/ABSOLUTE/REPO/rocm_tools/opencode/goal-adapter/index.mjs";
export default adaptGoal(plugin);
```

The adapter constrains the independent verifier's audit token and requirement
IDs to the current audit using JSON Schema enums. This prevents malformed IDs
from consuming its time limit. It preserves the Goal plugin's evidence checks,
tool restrictions, and authorization rules; it does not accept a verdict merely
because its IDs are valid.

Add the `goal` directory to the project's `plugins` list:

```json
{
  "package": "file:///ABSOLUTE/HOME/.local/share/opencode/rocm-exl3-plugins/goal",
  "options": {"autonomous": true}
}
```

The display plugin also accepts a `goalFormatter` option pointing to
`file:///.../node_modules/@bybrawe/opencode-goal/dist/tui/format.js` in its
**CLI** configuration to display the goal sidebar. `goalDirectory` can scope
that sidebar to one project.

Use goal commands explicitly; ordinary chat does not create a goal:

```text
/goal Fix the failing tests --check "python3 -m pytest tests -q"
/goal status
/goal pause
/goal resume
/goal budget --max-turns 20 --max-tokens 200000 --max-minutes 60
```

Goal state is project-local under `.opencode/goals`. Treat it as local state,
not source code. Completion uses the goal plugin's checks and verifier; a
status label alone is not proof that a task has been achieved.

## Validation

```bash
node --test rocm_tools/opencode/exl3-status/tests.mjs
node --test rocm_tools/opencode/goal-adapter/tests.mjs
python3 -m pytest rocm_tools/exl3_server/tests/test_metrics_cpu.py -q
```

HTTP tests cover metrics in ordinary JSON and final SSE chunks, including
clients that do not opt into usage chunks. Retired one-use tools remain valid
history while new calls must still be declared by the current request.
