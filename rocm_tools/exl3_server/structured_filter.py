"""Generation filters for structured OpenAI/Qwen output.

This module is intentionally independent of the HTTP server.  The parallel
tool filter stays active while it is waiting for a ``<tool_call>`` trigger, so
MTP verification advances its state after every accepted token.  Each call is
then constrained by a fresh LLGuidance matcher; whitespace between completed
Qwen XML blocks is accepted while argument values remain constrained.
"""

from __future__ import annotations

from typing import Any, Iterable


class ParallelToolCallFilter:
    """Always-active wrapper that constrains each triggered tool call.

    The wrapper deliberately exposes ``is_active=True`` even while waiting.
    This makes the generator refresh the all-allowed waiting mask after every
    accepted token, which is required when a trigger appears in an MTP verify
    window.  It does not constrain ordinary text or whitespace while waiting.
    """

    def __init__(
        self,
        tokenizer,
        *,
        call_grammar: str,
        trigger_token: int,
        eos_token_ids: Iterable[int] | None = None,
        max_calls: int = 8,
        required_first: bool = False,
        inner_factory=None,
    ):
        if inner_factory is None:
            from exllamav3.generator.filter import LLGuidanceFilter
            inner_factory = LLGuidanceFilter

        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.tokenizer = tokenizer
        self.trigger_token = None  # wrapper is always active
        self.call_trigger_token = int(trigger_token)
        self.eos_token_ids = {int(x) for x in (eos_token_ids or []) if x is not None}
        if not self.eos_token_ids:
            raise ValueError("parallel tool filter requires at least one EOS token")
        self.max_calls = int(max_calls)
        self.required_first = bool(required_first)
        self.is_active = True
        self.prefix_str = None
        self.eos_after_completed = False
        self.job = None
        self.generator = None
        self.vocab_size = None
        self._call_grammar = call_grammar
        self._filter_type = inner_factory
        self._inner = None
        self._state = "active" if self.required_first else "waiting"
        self._completed_calls = 0
        self._tokens: list[int] = []
        self._all_mask = None
        self._eos_mask = None
        self._new_inner()

    def _new_inner(self) -> None:
        self._inner = self._filter_type(
            self.tokenizer,
            trigger_token=None,
            eos_after_completed=False,
            llg_grammar=self._call_grammar,
        )

    def attach(self, job) -> None:
        self.job = job
        self.generator = job.generator
        self.vocab_size = job.generator.padded_vocab_size
        self._inner.attach(job)
        self._all_mask = None
        self._eos_mask = None

    def use_background_worker(self) -> bool:
        return True

    def reset(self) -> None:
        self._tokens.clear()
        self._completed_calls = 0
        self._state = "active" if self.required_first else "waiting"
        self.is_active = True
        self._inner.reset()
        self._inner.is_active = self._state == "active"

    def _start_call(self) -> None:
        self._inner.reset()
        self._inner.is_active = True
        self._state = "active"

    def _advance(self, token: int) -> bool:
        """Advance state for one already sampled token; return true on forced EOS."""
        token = int(token)
        if self._state == "done":
            return token in self.eos_token_ids
        if self._state == "waiting":
            if token == self.call_trigger_token and self._completed_calls < self.max_calls:
                self._start_call()
            return False

        self._inner.feed(token)
        if self._inner.is_completed():
            self._completed_calls += 1
            self._state = "done" if self._completed_calls >= self.max_calls else "waiting"
        return False

    def feed(self, token: int) -> bool:
        self._tokens.append(int(token))
        return self._advance(token)

    def rewind(self, num_tokens: int) -> None:
        if num_tokens < 0 or num_tokens > len(self._tokens):
            raise ValueError(f"cannot rewind {num_tokens} tokens from {len(self._tokens)}")
        if not num_tokens:
            return
        del self._tokens[-num_tokens:]
        self._completed_calls = 0
        self._state = "active" if self.required_first else "waiting"
        self._inner.reset()
        self._inner.is_active = self._state == "active"
        for token in self._tokens:
            self._advance(token)

    def _mask(self, allowed: Iterable[int] | None = None):
        import torch

        if self.vocab_size is None:
            raise RuntimeError("structured filter is not attached to a generation job")
        words = (self.vocab_size + 31) // 32
        out = torch.zeros((1, words), dtype=torch.int32)
        if allowed is None:
            out.fill_(-1)
            return out
        bits = out.view(-1).numpy().view("uint32")
        for token in allowed:
            token = int(token)
            if 0 <= token < self.vocab_size:
                bits[token >> 5] |= 1 << (token & 31)
        return out

    def get_next_logit_mask(self):
        if self._state == "active":
            return self._inner.get_next_logit_mask()
        if self._state == "done":
            if self._eos_mask is None:
                self._eos_mask = self._mask(self.eos_token_ids)
            return self._eos_mask
        if self._all_mask is None:
            self._all_mask = self._mask()
        return self._all_mask

    def is_completed(self) -> bool:
        return self._state == "done"

