"""Regressions for chunked prefill and the matched no-profiler control."""
import sys
import types
import unittest
from unittest.mock import patch
import torch
from rocm_tools.rdna2.profile_stages import NoopControls, run_timed_decode_job, release_gpu_resources


class FakeJob:
    def __init__(self, input_ids, max_new_tokens, **kwargs):
        self.input_ids = input_ids
        self.new_tokens = 0
        self.maximum = max_new_tokens
        self.sequences = [types.SimpleNamespace(sequence_ids=types.SimpleNamespace(
            torch=lambda: torch.cat((self.input_ids, torch.arange(
                100, 100 + self.new_tokens, dtype=torch.long).reshape(1, -1)), dim=1)))]


class FakeGenerator:
    def __init__(self):
        self.job = None
        self.calls = 0

    def enqueue(self, job):
        self.job = job

    def num_remaining_jobs(self):
        return int(self.job is not None and self.job.new_tokens < self.job.maximum)

    def iterate(self):
        self.calls += 1
        if self.calls < 4:  # three prefill chunks before the first output token
            return []
        self.job.new_tokens += 1
        if self.job.new_tokens == self.job.maximum:
            return [{'stage': 'streaming', 'eos': True,
                     'prompt_tokens': self.job.input_ids.numel(), 'cached_tokens': 0,
                     'new_tokens': self.job.maximum, 'eos_reason': 'max_new_tokens',
                     'time_prefill': 0.01, 'time_generate': 0.02}]
        return []


class StageControlTests(unittest.TestCase):
    def test_pre_import_failure_cleanup_does_not_import_engine(self):
        errors = []
        with patch.dict(sys.modules, {'exllamav3': None}):
            release_gpu_resources(None, {}, errors)
        self.assertEqual(errors, [])

    def exercise(self, fail_sync=False):
        sampler = types.ModuleType('exllamav3.generator.sampler')
        sampler.ArgmaxSampler = object
        calls = []

        def sync():
            calls.append(1)
            if fail_sync and len(calls) == 2:
                raise RuntimeError('synthetic end-sync failure')

        with patch.dict(sys.modules, {'exllamav3.generator.sampler': sampler}):
            result = run_timed_decode_job(
                FakeGenerator(), FakeJob, torch.zeros((1, 8192), dtype=torch.long),
                8192, 1, 1234, NoopControls(), 4, 3, sync)
        return result, calls

    def test_chunked_prefill_and_2d_sequence(self):
        result, _ = self.exercise()
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['prefill_window']['zero_iters'], 3)
        self.assertEqual(result['sequence']['sequence_len'], 8199)
        self.assertEqual(result['sequence']['generated_ids'], list(range(100, 107)))
        self.assertEqual(result['decode_window']['events_n'], 3)

    def test_control_keeps_boundary_sync_and_wall_clocks(self):
        result, calls = self.exercise()
        self.assertEqual(len(calls), 4)
        for name in ('prefill_window', 'decode_window'):
            self.assertFalse(result[name]['roctx_traced'])
            self.assertGreater(result[name]['wall_s'], 0)

    def test_window_sync_failure_cannot_report_success(self):
        result, _ = self.exercise(fail_sync=True)
        self.assertTrue(any('prefill window teardown' in x for x in result['problems']))


if __name__ == '__main__':
    unittest.main()
