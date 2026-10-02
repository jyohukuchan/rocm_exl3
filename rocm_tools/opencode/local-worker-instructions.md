# Local Qwen workers

The primary agent and the `worker` subagent use the local Qwen model with xhigh reasoning.

When the user asks for delegation, use OpenCode V2's **direct native `subagent` tool**, selecting `worker`. It is not in the Code Mode catalog: do not call `tools.subagent` inside `execute` or search that catalog for it. Use the actual advertised native tool schema.

For independent work, the primary agent can send up to three native subagent calls in the same turn and wait for their results. Assign each worker a bounded responsibility and its own files. Workers share the tree with other agents: preserve their changes and adapt to them. Workers should complete their assignment directly rather than launching more workers.

The inference server admits at most three requests at once, including primary-agent requests. Further requests wait. Its 768Ki KV pool is shared across all active jobs; it is not 768Ki for each worker. Keep delegated prompts and returned reports focused, and state whether the assignment is read-only or permits edits. Report child failures honestly.
