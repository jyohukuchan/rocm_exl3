#!/usr/bin/env python3
"""CPU-only tests for the hybrid recurrent (Qwen4Exp: GDN + PLE) support in
collect_top1.py -- a bounded Phase-5 prerequisite for Qwen3.8 Flash Next.

Nothing here touches a GPU, exllamav3 or torch: the chunked recurrent route is
driven through the REAL orchestration code (collect_top1.run_chunked_case)
against behavior fakes that enforce the same engine contract as
recurrent_util.prepare_for_recurrence / advance_recurrent_states and
PLELayer._state_history:

  * params for every chunk past position 0 must carry the SAME live state whose
    .position equals past_len -- each chunk from a zero state trips the fake's
    assert (the "never silently restart each chunk from zero" property),
  * the PLE hashing history a chunk sees is cat(carried window, chunk ids) and
    must equal the literal EOS-padded prefix history the stateless bulk route
    builds (ple_reference_history), across every chunk boundary,
  * state advancement, per-case slot release (also when a forward raises),
    pool-reset barriers between independent cases, and release failures being
    RECORDED (never swallowed, never masking the original error).

The n-gram table residency summary (--ngram-ram) and the CLI cross-checks are
pure and tested directly. Fixtures are input data and hand-checked expected
numbers, not implementation text matching. These tests prove the collector's
ORCHESTRATION; they are NOT evidence that any real model ran -- the root
orchestrator tests the actual Qwen3.8 Flash Next model after download.

Run from the repo root (or anywhere):
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2 import collect_top1


CTX = 3          # PLE hashing context (ngram_size - 1)
EOS = 0          # PLE eos token id used for sequence-start padding
VOCAB = 97


# ---------------------------------------------------------------------------
# behavior fakes: cache slot pool + engine-contract-enforcing chunk forward
# ---------------------------------------------------------------------------

class FakeState:
    """Stands in for GDNState: live slot handle with the engine's .position
    contract, plus the PLE id-history window PLELayerState.clear() initializes
    to full EOS (the EOS-padded sequence start)."""

    def __init__(self, cache, slot):
        self.cache = cache
        self.slot = slot
        self.position = 0
        self.last_history = 0
        self.window = [EOS] * CTX      # cleared slot: eos-filled history context
        self.freed = False

    def free(self):
        self.cache.release_state(self)


class FakeCache:
    """Cache.get_new_state/release_state/reset_states pool semantics."""

    def __init__(self, num_slots=2):
        self.num_slots = num_slots
        self.pool = list(range(num_slots))
        self.gets = 0
        self.resets = 0
        self._next_slot = 0

    def get_new_state(self):
        assert self.pool, "Cannot create new state: no available slots"
        self.pool.remove(self.pool[0])
        self.gets += 1
        st = FakeState(self, self._next_slot % self.num_slots)
        self._next_slot += 1
        return st

    def release_state(self, state):
        state.freed = True            # double release is observable via this flag

    def reset_states(self):
        self.resets += 1
        self.pool = list(range(self.num_slots))


class FakeEngine:
    """Fake cached recurrent chunk forward. Enforces (AssertionError) the same
    contract the exl3 engine does, and returns fake logits rows the driver
    harvests:
      - state.position must equal past_len BEFORE the forward (prepare_for_recurrence),
      - PLE history = cat(carried window, chunk ids) must equal the literal
        EOS-padded reference prefix history for these exact global positions
        (PLELayer._state_history vs the stateless bulk _history),
      - after the forward the state advances by seqlen (advance_recurrent_states)
        and the PLE window becomes the trailing CTX entries of that history
        (id_state[slot, :ctx] = history[-ctx:]), unless stalled for testing.

    modes: "stall" never advances position; "boom_at" raises inside that chunk;
    "restart" re-zeroes the carry each chunk (the exact silent failure the
    contract must catch: every chunk recomputing from a fresh state).
    """

    def __init__(self, ids, nonfinite_positions=(), mode="ok", boom_at=None):
        self.ids = ids
        self.ref = collect_top1.ple_reference_history(ids, CTX, EOS)
        self.nonfinite = set(nonfinite_positions)
        self.mode = mode
        self.boom_at = boom_at
        self.states_seen = []          # distinct state objects across chunks
        self.chunk_windows = []        # history each chunk hashed with
        self.n_calls = 0

    def forward_chunk(self, s, e, params):
        self.n_calls += 1
        rs = params["recurrent_states"]
        st = rs[0] if rs else None
        assert params["past_len"] == s
        if st is not None:
            assert st.position == s, "recurrent states don't match input past_len"
            self.states_seen.append(st)
            if self.mode == "restart":
                st.window = [EOS] * CTX  # simulate each chunk running from a zero state
            history = list(st.window) + self.ids[s:e]
            reference = self.ref[s : s + CTX + (e - s)]
            assert history == reference, (
                f"PLE history at chunk [{s},{e}) is not the real preceding-token "
                f"context: {history} != {reference}")
            self.chunk_windows.append(history)
            st.window = collect_top1.ple_carry_update(st.window, self.ids[s:e], CTX, EOS)
            if self.mode != "stall":
                st.position += e - s   # advance_recurrent_states
        if self.boom_at == s:
            raise RuntimeError(f"boom in chunk [{s},{e})")
        return [{"pos": p,
                 "top1": (p * 7 + 1) % VOCAB,
                 "finite": p not in self.nonfinite}
                for p in range(s, e)]

    @staticmethod
    def rows_of(logits, s, e):
        return [(row["pos"], row) for row in logits]

    @staticmethod
    def harvest_row(row):
        return row["top1"], row["finite"]

    def params_for_chunk(self, s, state):
        return {"attn_mode": "flash_attn", "cache": None, "past_len": s,
                "batch_shape": (1, 4096),
                "recurrent_states": [state] if state is not None else None}


def make_case(cid, ids, positions=None):
    return {"case_id": cid, "ids": ids, "len_ids": len(ids),
            "positions": positions if positions is not None else list(range(len(ids)))}


def run_recurrent(case, cache, engine, chunk=4, reset_pool=True,
                  verify_carry=None, free_state=None):
    ev = {}
    top1, nonfinite = collect_top1.run_chunked_case(
        case, chunk, recurrent=True,
        forward_chunk=engine.forward_chunk,
        rows_of=engine.rows_of,
        harvest_row=engine.harvest_row,
        params_for_chunk=engine.params_for_chunk,
        new_state=cache.get_new_state,
        free_state=free_state if free_state is not None else lambda st: st.free(),
        reset_pool=cache.reset_states if reset_pool else None,
        verify_carry=verify_carry,
        evidence=ev,
    )
    return top1, nonfinite, ev


# ---------------------------------------------------------------------------
# state carry / no-zero-restart / advancement
# ---------------------------------------------------------------------------

class StateCarryTests(unittest.TestCase):
    def test_one_live_state_carried_across_every_chunk(self):
        ids = list(range(30, 30 + 10))
        cache, engine = FakeCache(), FakeEngine(ids)
        case = make_case("a", ids)
        top1, nonfinite = collect_top1.run_chunked_case(
            case, 4, True, engine.forward_chunk, engine.rows_of,
            engine.harvest_row, engine.params_for_chunk,
            new_state=cache.get_new_state, free_state=lambda st: st.free(),
            reset_pool=cache.reset_states)
        self.assertEqual(engine.n_calls, 3)               # chunks [0,4)[4,8)[8,10)
        self.assertEqual(cache.gets, 1)                  # ONE state for the case
        self.assertTrue(all(st is engine.states_seen[0] for st in engine.states_seen))
        self.assertEqual(engine.states_seen[0].position, len(ids))
        self.assertEqual(top1, [(p * 7 + 1) % VOCAB for p in range(len(ids))])
        self.assertEqual(nonfinite, [])

    def test_missing_positions_still_raise_after_carry(self):
        ids = list(range(5))
        cache, engine = FakeCache(), FakeEngine(ids)
        case = make_case("a", ids, positions=[0, 4, 9])   # 9 can never be produced
        with self.assertRaises(RuntimeError) as cm:
            run_recurrent(case, cache, engine)
        self.assertIn("no logits for positions", str(cm.exception))
        self.assertIn("9", str(cm.exception))
        # error path cleanup still ran
        self.assertEqual(engine.states_seen[0].freed, True)
        self.assertEqual(cache.resets, 1)

    def test_stalled_engine_state_fails_closed(self):
        """A forward that consumed a chunk but did not advance the state would
        make every later chunk a silent zero-history recompute; the driver must
        refuse, not harvest on."""
        ids = list(range(20, 40))
        cache, engine = FakeCache(), FakeEngine(ids, mode="stall")
        case = make_case("a", ids)
        with self.assertRaises(RuntimeError) as cm:
            run_recurrent(case, cache, engine)
        self.assertIn("did not advance", str(cm.exception))
        self.assertEqual(engine.n_calls, 1)              # stopped at the first chunk
        self.assertTrue(engine.states_seen[0].freed)     # released anyway
        self.assertEqual(cache.resets, 1)

    def test_new_state_not_at_position_zero_refused(self):
        cache, engine = FakeCache(), FakeEngine([1, 2, 3])
        st = FakeState(cache, 0)
        st.position = 7
        with self.assertRaises(RuntimeError) as cm:
            collect_top1.run_chunked_case(
                make_case("a", [1, 2, 3]), 2, True,
                engine.forward_chunk, engine.rows_of, engine.harvest_row,
                engine.params_for_chunk, new_state=lambda: st,
                free_state=lambda s: s.free(), reset_pool=cache.reset_states)
        self.assertIn("starts at position 7", str(cm.exception))
        self.assertTrue(st.freed)

    def test_nonrecurrent_route_untouched(self):
        """recurrent=False must behave exactly like the pre-existing chunked
        collector: no state pool calls, recurrent_states stays None."""
        ids = list(range(8))
        engine = FakeEngine(ids)
        calls = {"states": 0}

        def bad_new_state():
            calls["states"] += 1
            raise AssertionError("nonrecurrent run must not allocate recurrent state")

        top1, nonfinite = collect_top1.run_chunked_case(
            make_case("a", ids), 3, False,
            engine.forward_chunk, engine.rows_of, engine.harvest_row,
            lambda s, state: {"attn_mode": "flash_attn", "past_len": s,
                              "batch_shape": (1, 4096),
                              "recurrent_states": [state] if state is not None else None},
            new_state=bad_new_state, free_state=lambda st: st.free(),
            reset_pool=lambda: None)
        self.assertEqual(calls["states"], 0)
        self.assertEqual(top1, [(p * 7 + 1) % VOCAB for p in range(8)])
        self.assertEqual(nonfinite, [])

    def test_recurrent_without_lifecycle_callbacks_refused(self):
        engine = FakeEngine([1, 2, 3])
        with self.assertRaises(RuntimeError):
            collect_top1.run_chunked_case(
                make_case("a", [1, 2, 3]), 2, True,
                engine.forward_chunk, engine.rows_of, engine.harvest_row,
                engine.params_for_chunk)


# ---------------------------------------------------------------------------
# PLE token history: EOS-padded start, real context across chunk boundaries
# ---------------------------------------------------------------------------

class PleHistoryTests(unittest.TestCase):
    def test_engine_seen_history_matches_eos_padded_reference(self):
        """The FakeEngine asserts on every chunk that the hashing history is
        exactly ref[s : s+ctx+chunk_len] of [eos]*CTX + ids; passing with a
        chunk size SMALLER than the context (carry must span multiple chunks)
        and with EOS token ids appearing as literal content mid-sequence proves
        both the sequence-start padding and the real-context boundary carry."""
        ids = [5, EOS, 7, 8, EOS, 9, EOS, 11, 12]
        cache, engine = FakeCache(), FakeEngine(ids)
        top1, _, _ = run_recurrent(make_case("a", ids), cache, engine, chunk=2)
        self.assertEqual(len(engine.chunk_windows), 5)
        ref = collect_top1.ple_reference_history(ids, CTX, EOS)
        # first chunk hashes with the EOS-padded sequence start
        self.assertEqual(engine.chunk_windows[0], ref[0 : CTX + 2])
        # a boundary chunk's carry is the REAL preceding-token window (the
        # literal ids[1:4], eos token included as content), not an eos re-pad
        self.assertEqual(engine.chunk_windows[2], ref[4 : 4 + CTX + 2])
        self.assertEqual(engine.chunk_windows[2][:CTX], ids[1:4])
        self.assertEqual(top1[4], (4 * 7 + 1) % VOCAB)

    def test_zero_restarted_chunk_detected(self):
        """The exact silent failure to rule out: every chunk recomputing from a
        fresh state. Chunk 0 (EOS start) still matches, but at the first
        boundary the real preceding-token context is gone -- the engine
        contract fake rejects it and the driver releases the state anyway."""
        ids = list(range(50, 60))
        cache, engine = FakeCache(), FakeEngine(ids, mode="restart")
        with self.assertRaises(AssertionError) as cm:
            run_recurrent(make_case("a", ids), cache, engine, chunk=4)
        self.assertIn("preceding-token", str(cm.exception))
        self.assertEqual(engine.n_calls, 2)              # passed chunk 0, failed at boundary
        self.assertTrue(engine.states_seen[-1].freed)    # cleanup on the error path
        self.assertEqual(cache.resets, 1)

    def test_carry_update_chain_equals_bulk_reference_property(self):
        """Pure property: chaining ple_carry_update over ANY chunking of ANY ids
        reproduces the exact stateless bulk history windows (the equivalence the
        chunked-vs-bulk comparison relies on for input identity)."""
        ids = [4, EOS, 6, 7, EOS, 9, 10, EOS, 12, 13, 14]
        ref = collect_top1.ple_reference_history(ids, CTX, EOS)
        for plan in ([1] * len(ids), [2, 3, 6], [len(ids)], [7, 1, 3]):
            carry = None
            s = 0
            for c in plan:
                e = min(s + c, len(ids))
                windowed = (carry if carry is not None else [EOS] * CTX) + ids[s:e]
                self.assertEqual(windowed, ref[s : s + CTX + (e - s)],
                                 f"chunk plan {plan} diverges at [{s},{e})")
                carry = collect_top1.ple_carry_update(carry, ids[s:e], CTX, EOS)
                self.assertEqual(carry, ref[e : e + CTX])
                s = e
            self.assertEqual(s, len(ids))

    def test_short_sequence_keeps_eos_padding_in_carry(self):
        # sequence shorter than the context: carry stays eos-padded at the FRONT
        carry = collect_top1.ple_carry_update(None, [42], CTX, EOS)
        self.assertEqual(carry, [EOS, EOS, 42])
        carry2 = collect_top1.ple_carry_update(carry, [43], CTX, EOS)
        self.assertEqual(carry2, [EOS, 42, 43])

    def test_zero_context_edge(self):
        self.assertEqual(collect_top1.ple_carry_update(None, [1, 2], 0, EOS), [])
        self.assertEqual(collect_top1.ple_reference_history([1, 2], 0, EOS), [1, 2])


# ---------------------------------------------------------------------------
# per-case reset / release / error cleanup
# ---------------------------------------------------------------------------

class StateLifecycleCleanupTests(unittest.TestCase):
    def test_independent_cases_get_fresh_state_and_return_the_slot(self):
        cache = FakeCache(num_slots=1)       # second case fails if slot not returned
        for i, ids in enumerate((list(range(7)), list(range(9)))):
            engine = FakeEngine(ids)
            top1, _, ev = run_recurrent(make_case(f"c{i}", ids), cache, engine, chunk=3)
            self.assertEqual(ev["states_created"], 1)
            self.assertEqual(ev["states_released"], 1)
            self.assertEqual(ev["release_failures"], [])
            self.assertEqual(cache.resets, i + 1)   # pool barrier per case
            self.assertEqual(len(top1), len(ids))
        self.assertEqual(cache.gets, 2)

    def test_forward_exception_midcase_releases_state(self):
        ids = list(range(12))
        cache = FakeCache(num_slots=1)
        engine = FakeEngine(ids, boom_at=4)   # raises in the SECOND chunk
        with self.assertRaises(RuntimeError) as cm:
            run_recurrent(make_case("a", ids), cache, engine, chunk=4)
        self.assertIn("boom", str(cm.exception))
        self.assertEqual(cache.resets, 1)
        # pool was rebuilt and the state flagged freed -> next case can run
        engine2 = FakeEngine(ids)
        top1, _, _ = run_recurrent(make_case("b", ids), cache, engine2, chunk=4)
        self.assertEqual(top1[0], 1)

    def test_release_failure_recorded_not_raised_and_reset_still_runs(self):
        ids = list(range(6))
        cache = FakeCache()
        engine = FakeEngine(ids)

        def bad_free(state):
            raise RuntimeError("slot release exploded")

        _, _, ev = run_recurrent(make_case("a", ids), cache, engine,
                                 chunk=3, free_state=bad_free)
        self.assertEqual(len(ev["release_failures"]), 1)
        self.assertIn("slot release exploded", ev["release_failures"][0])
        self.assertEqual(ev["states_released"], 0)
        self.assertEqual(cache.resets, 1)     # barrier ran despite the bad free

    def test_release_failure_does_not_mask_case_error(self):
        ids = list(range(6))
        cache = FakeCache()
        engine = FakeEngine(ids, boom_at=3)

        def bad_free(state):
            raise RuntimeError("release masked?")

        with self.assertRaises(RuntimeError) as cm:
            run_recurrent(make_case("a", ids), cache, engine, chunk=3,
                          free_state=bad_free)
        self.assertIn("boom", str(cm.exception))      # original error survives
        # (cleanup evidence dict isn't returned on the error path, but the
        # finally chain ran: reset happened once)
        self.assertEqual(cache.resets, 1)

    def test_verify_carry_note_recorded_and_raise_propagates(self):
        ids = list(range(10))

        def note_verify(state, e):
            return f"not externally readable at {e}"

        cache, engine = FakeCache(), FakeEngine(ids)
        _, _, ev = run_recurrent(make_case("a", ids), cache, engine, chunk=4,
                                 verify_carry=note_verify)
        self.assertEqual(len(ev["carry_notes"]), 3)    # one per boundary
        self.assertIn("not externally readable at 4", ev["carry_notes"][0])

        def bad_verify(state, e):
            raise RuntimeError("PLE history desync")

        cache2 = FakeCache()
        engine2 = FakeEngine(ids)
        with self.assertRaises(RuntimeError):
            run_recurrent(make_case("b", ids), cache2, engine2,
                          chunk=4, verify_carry=bad_verify)
        self.assertEqual(engine2.n_calls, 1)             # raised at the 1st boundary
        self.assertEqual(cache2.resets, 1)

    def test_runtime_verify_carry_recomputes_expected_window_from_manifest_ids(self):
        """The REAL closure-style verify (same computation collect_exl3 wires):
        re-deriving the expected carry with ple_carry_update(None, ids[:e]) must
        equal the fake state's carried window at every boundary."""
        ids = [3, EOS, 5, 6, 7, EOS, 9]
        cache, engine = FakeCache(), FakeEngine(ids)

        def verify(state, e):
            expected = collect_top1.ple_carry_update(None, ids[:e], CTX, EOS)
            assert state.window == expected, (state.window, expected)
            return None

        run_recurrent(make_case("a", ids), cache, engine, chunk=2, verify_carry=verify)
        self.assertEqual(engine.n_calls, 4)


# ---------------------------------------------------------------------------
# speculative/MTP exclusion + params hygiene
# ---------------------------------------------------------------------------

class NoSpeculationTests(unittest.TestCase):
    def test_recurrent_history_param_never_tolerated(self):
        ids = list(range(6))
        cache = FakeCache()
        engine = FakeEngine(ids)

        def params_with_history(s, state):
            p = engine.params_for_chunk(s, state)
            p["recurrent_history"] = True      # simulate a spec-decoding leak
            return p

        with self.assertRaises(RuntimeError) as cm:
            collect_top1.run_chunked_case(
                make_case("a", ids), 3, True,
                engine.forward_chunk, engine.rows_of, engine.harvest_row,
                params_with_history, new_state=cache.get_new_state,
                free_state=lambda st: st.free(), reset_pool=cache.reset_states)
        self.assertIn("recurrent_history", str(cm.exception))
        self.assertEqual(engine.n_calls, 0)    # refused before ANY forward
        self.assertEqual(cache.resets, 1)


# ---------------------------------------------------------------------------
# non-finite rows and position mapping through the chunked recurrent route
# ---------------------------------------------------------------------------

class HarvestMappingTests(unittest.TestCase):
    def test_positions_subset_ordering_and_nonfinite_flags(self):
        ids = list(range(10))
        cache = FakeCache()
        engine = FakeEngine(ids, nonfinite_positions=(2, 5))
        case = make_case("a", ids, positions=[2, 5, 9])
        top1, nonfinite = collect_top1.run_chunked_case(
            case, 4, True, engine.forward_chunk, engine.rows_of,
            engine.harvest_row, engine.params_for_chunk,
            new_state=cache.get_new_state, free_state=lambda st: st.free(),
            reset_pool=cache.reset_states)
        self.assertEqual(top1, [(2 * 7 + 1) % VOCAB, (5 * 7 + 1) % VOCAB,
                                (9 * 7 + 1) % VOCAB])
        self.assertEqual(sorted(nonfinite), [2, 5])   # flagged but still recorded


# ---------------------------------------------------------------------------
# n-gram table residency: recorded modes + refusal of a wrong residence
# ---------------------------------------------------------------------------

class NGramResidencyTests(unittest.TestCase):
    def test_ram_requested_and_delivered(self):
        rep, err = collect_top1.ngram_residency_report(
            {"a.ple": "trellis_ram", "b.ple": "fp16_ram"}, ram_requested=True)
        self.assertIsNone(err)
        self.assertEqual(rep["residency"], "ram")
        self.assertEqual(rep["counts"], {"total": 2, "ram": 2, "disk": 0, "other": 0})

    def test_ram_requested_disk_delivered_refused(self):
        rep, err = collect_top1.ngram_residency_report(
            {"a.ple": "trellis_disk"}, ram_requested=True)
        self.assertIsNotNone(err)
        self.assertIn("--ngram-ram", err)
        self.assertEqual(rep["residency"], "disk")

    def test_ram_requested_mixed_or_unknown_refused(self):
        _, err = collect_top1.ngram_residency_report(
            {"a": "fp16_ram", "b": "trellis_disk"}, ram_requested=True)
        self.assertIsNotNone(err)
        _, err2 = collect_top1.ngram_residency_report(
            {"a": "fp16_ram", "b": None}, ram_requested=True)   # never loaded -> unknown
        self.assertIsNotNone(err2)
        _, err3 = collect_top1.ngram_residency_report({}, ram_requested=True)
        self.assertIsNotNone(err3)          # request vacuous: refuse, don't silently pass

    def test_no_request_records_without_refusing(self):
        rep, err = collect_top1.ngram_residency_report(
            {"a": "trellis_disk"}, ram_requested=False)
        self.assertIsNone(err)
        self.assertEqual(rep["tables"], {"a": "trellis_disk"})
        self.assertEqual(rep["residency"], "disk")
        rep2, err2 = collect_top1.ngram_residency_report({}, ram_requested=False)
        self.assertIsNone(err2)
        self.assertEqual(rep2["residency"], "none")


class NGramModuleWalkTests(unittest.TestCase):
    class Table:
        def __init__(self, key, mode, children=()):
            self.key = key
            self.mode = mode
            self.modules = list(children)

    class Wrapper:
        def __init__(self, children):
            self.modules = list(children)

    def setUp(self):
        # exact class NAME match is what the duck-typed walk keys on
        self.T = type("NGramEmbedding", (), {})

    def test_walks_nested_and_collects_modes(self):
        t1 = self.T(); t1.key = "layers.0.ple.ple_embedding"; t1.mode = "trellis_disk"; t1.modules = []
        t2 = self.T(); t2.key = "layers.5.ple.ple_embedding"; t2.mode = "trellis_ram"; t2.modules = []
        decoy = self.Table("x", "not_a_table")       # wrong class name -> ignored
        model = self.Wrapper([self.Wrapper([t1, decoy]), t2])
        got = collect_top1.collect_ngram_table_modes(model)
        self.assertEqual(got, {t1.key: "trellis_disk", t2.key: "trellis_ram"})

    def test_walk_survives_cycles(self):
        t = self.T(); t.key = "t"; t.mode = "fp16_ram"
        w = self.Wrapper([t]); t.modules = [w]       # cycle
        self.assertEqual(collect_top1.collect_ngram_table_modes(self.Wrapper([w])),
                         {"t": "fp16_ram"})


# ---------------------------------------------------------------------------
# bulk route stays stateless BY CONSTRUCTION + CLI wiring
# ---------------------------------------------------------------------------

class BulkAndCliTests(unittest.TestCase):
    def test_bulk_params_carry_no_cache_no_batch_shape_no_states(self):
        """prepare_for_recurrence is a no-op exactly when batch_shape,
        cache_seqlens and recurrent_states are all absent -- that IS the
        stateless-full-sequence semantics the bulk route relies on for
        recurrent models (GDN zero-init chunk scan, PLE [eos]*ctx + ids)."""
        p = collect_top1.make_bulk_params()
        self.assertEqual(p["attn_mode"], "flash_attn_nc")
        for key in ("cache", "batch_shape", "cache_seqlens", "recurrent_states"):
            self.assertNotIn(key, p)

    def test_ngram_ram_cli(self):
        parser = collect_top1.build_parser()
        base = ["--manifest", "m", "-m", "d", "--backend", "exl3", "-o", "o"]
        args = parser.parse_args(base + ["--ngram-ram"])
        self.assertTrue(args.ngram_ram)
        args2 = parser.parse_args(base)
        self.assertFalse(args2.ngram_ram)

    def test_ngram_ram_refused_on_transformers(self):
        ns = argparse.Namespace(backend="transformers", use_per_device=None,
                                cache_tokens=None, ngram_ram=True)
        with self.assertRaises(SystemExit) as cm:
            collect_top1.check_cli_options(ns)
        self.assertIn("--ngram-ram", str(cm.exception))

    def test_legacy_namespaces_without_ngram_ram_still_pass(self):
        # test_ls_split_cpu builds Namespaces lacking this attr; check must not break
        ns = argparse.Namespace(backend="transformers", use_per_device=None,
                                cache_tokens=None)
        self.assertIsNone(collect_top1.check_cli_options(ns))


if __name__ == "__main__":
    unittest.main(verbosity = 2)
