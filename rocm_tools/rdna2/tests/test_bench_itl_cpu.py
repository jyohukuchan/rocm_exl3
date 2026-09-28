#!/usr/bin/env python3
"""CPU-only tests for bench.py's per-token ITL helpers (no GPU, no torch).

Covers percentile() with hand-checked known numbers (nearest-rank, the result
is always an actually observed sample) and token_intervals() event-stream
semantics: the first token anchors prefill/TTFT and yields no sample, plain
+1 increments give N-1 positive intervals for N tokens, and multi-token jumps
are reported as bursts with NO averaged fake ITL sample. bench.py imports
torch/exllamav3 only inside functions, so importing it here is host-safe.

Run from the repo root (or anywhere):
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2 import bench


class TestPercentile(unittest.TestCase):
    def test_odd_count_median_and_p95_are_observed_samples(self):
        # 10 samples 10..100: nearest-rank p50 -> ceil(5)=5th -> 50;
        # p95 -> ceil(9.5)=10th -> 100.
        samples = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        self.assertEqual(bench.percentile(samples, 50.0), 50.0)
        self.assertEqual(bench.percentile(samples, 95.0), 100.0)

    def test_even_count_ranks(self):
        # 20 samples 1..20: p50 -> ceil(10)=10th -> 10.0 exactly (the 0.95
        # float artifact must NOT push the p95 rank from 19 to 20).
        samples = [float(i) for i in range(1, 21)]
        self.assertEqual(bench.percentile(samples, 50.0), 10.0)
        self.assertEqual(bench.percentile(samples, 95.0), 19.0)

    def test_unsorted_input_is_ranked_not_positional(self):
        samples = [3.0, 1.0, 2.0]
        self.assertEqual(bench.percentile(samples, 50.0), 2.0)
        self.assertEqual(bench.percentile(samples, 95.0), 3.0)

    def test_single_sample(self):
        self.assertEqual(bench.percentile([42.5], 50.0), 42.5)
        self.assertEqual(bench.percentile([42.5], 95.0), 42.5)

    def test_smaller_than_nearest_rank(self):
        # n=255 (decode 512x256 style): p95 -> ceil(242.25)=243rd smallest.
        samples = [float(i) for i in range(255)]
        self.assertEqual(bench.percentile(samples, 95.0), 242.0)
        self.assertEqual(bench.percentile(samples, 50.0), 127.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            bench.percentile([], 50.0)

    def test_q_out_of_range_raises(self):
        for q in (0.0, -5.0, 100.5):
            with self.assertRaises(ValueError):
                bench.percentile([1.0, 2.0], q)
        self.assertEqual(bench.percentile([1.0, 2.0], 100.0), 2.0)


class TestTokenIntervals(unittest.TestCase):
    def test_plain_ar_gives_n_minus_1_intervals_and_no_bursts(self):
        # 5 tokens, +1 per iterate(); binary-exact times so ms values are exact.
        events = [(0.0, 1), (0.25, 2), (0.5, 3), (1.0, 4), (1.5, 5)]
        samples, bursts = bench.token_intervals(events)
        self.assertEqual(bursts, [])
        self.assertEqual(samples, [250.0, 250.0, 500.0, 500.0])  # first dt excluded

    def test_single_token_yields_no_intervals(self):
        samples, bursts = bench.token_intervals([(0.0, 1)])
        self.assertEqual(samples, [])
        self.assertEqual(bursts, [])

    def test_no_events(self):
        self.assertEqual(bench.token_intervals([]), ([], []))

    def test_multi_token_jump_is_a_burst_not_an_averaged_sample(self):
        events = [(0.0, 1), (0.5, 3), (0.75, 4)]
        samples, bursts = bench.token_intervals(events)
        # The 0.5s for 2 tokens is NOT emitted as two fake 250ms ITL samples.
        self.assertEqual(samples, [250.0])
        self.assertEqual(len(bursts), 1)
        self.assertEqual(bursts[0]["from_new_tokens"], 1)
        self.assertEqual(bursts[0]["to_new_tokens"], 3)
        self.assertEqual(bursts[0]["delta"], 2)
        self.assertAlmostEqual(bursts[0]["dt_ms"], 500.0)

    def test_leading_burst_anchor_keeps_subsequent_itls(self):
        events = [(0.0, 2), (0.5, 3)]
        samples, bursts = bench.token_intervals(events)
        self.assertEqual(len(bursts), 1)
        self.assertEqual(bursts[0]["delta"], 2)
        self.assertIsNone(bursts[0]["dt_ms"])
        self.assertEqual(samples, [500.0])

    def test_nonpositive_intervals_excluded(self):
        events = [(0.0, 1), (0.0, 2), (0.25, 3)]
        samples, bursts = bench.token_intervals(events)
        self.assertEqual(samples, [250.0])   # zero dt dropped, not a sample
        self.assertEqual(bursts, [])

    def test_count_matches_new_tokens_minus_one_for_plain_ar(self):
        # Property check over a simulated full AR run: 300 tokens -> 299 samples.
        events = [((i + 1) * 0.001, i + 1) for i in range(300)]
        samples, bursts = bench.token_intervals(events)
        self.assertEqual(bursts, [])
        self.assertEqual(len(samples), 299)
        self.assertTrue(all(s > 0.0 for s in samples))


class ObservedTimingTests(unittest.TestCase):
    def test_slow_first_token_is_not_steady_decode_time(self):
        timing = bench.observed_timing([(150.0, 1), (151.0, 2), (153.0, 3)], 100.0)
        self.assertEqual(timing["first_token_wall_ms"], 50000.0)
        self.assertEqual(timing["decode_observed_s"], 3.0)
        self.assertAlmostEqual(timing["decode_observed_tps"], 2 / 3)

    def test_prefill_only_has_no_decode_rate(self):
        timing = bench.observed_timing([(2.0, 1)], 1.0)
        self.assertEqual(timing["first_token_wall_ms"], 1000.0)
        self.assertIsNone(timing["decode_observed_tps"])

    def test_missing_or_invalid_first_event_is_refused(self):
        for events in ([], [(1.0, 1)]):
            with self.assertRaises(ValueError):
                bench.observed_timing(events, 1.0)


if __name__ == "__main__":
    unittest.main()
