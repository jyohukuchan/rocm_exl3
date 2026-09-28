"""Regressions for chunked prefill, the matched no-profiler control, the
two-GPU layer-split wiring (budgets, load kwargs, ALL-device boundary sync,
placement-driven cleanup) and the opt-in transfer observer's stage_window
integration (arm after the opening sync, snapshot after the closing sync,
failures recorded instead of swallowed, no "transfers" key when OFF)."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rocm_tools.rdna2.profile_stages import (
    NoopControls,
    MAX_CHUNK_SIZE,
    make_boundary_sync,
    model_load_kwargs,
    parse_use_per_device,
    release_gpu_resources,
    run_timed_decode_job,
    stage_window,
)


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

    def exercise(self, fail_sync=False, observer=None):
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
                8192, 1, 1234, NoopControls(), 4, 3, sync, observer=observer)
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
            self.assertNotIn('transfers', result[name])   # observer OFF: identical schema

    def test_window_sync_failure_cannot_report_success(self):
        result, _ = self.exercise(fail_sync=True)
        self.assertTrue(any('prefill window teardown' in x for x in result['problems']))


class FakeCudaForSync:
    def __init__(self):
        self.log = []

    def synchronize(self, dev=None):
        self.log.append(dev)

    def empty_cache(self):
        self.log.append('empty_cache')


class FakeTorchForSync:
    def __init__(self):
        self.cuda = FakeCudaForSync()

    @staticmethod
    def device(spec):
        return str(spec)


class SplitWiringTests(unittest.TestCase):
    def test_budgets_delegated_to_multi_gpu_validator(self):
        self.assertIsNone(parse_use_per_device(None))
        self.assertEqual(parse_use_per_device([3, 4]), [3.0, 4.0])
        self.assertEqual(parse_use_per_device([6.5, 8]), [6.5, 8.0])

    def test_budgets_reject_short_zero_nonfinite(self):
        for bad in ([3], [3, 0], [3, -1], [float('inf')], [3, float('nan')]):
            with self.assertRaises(SystemExit, msg=bad):
                parse_use_per_device(bad)

    def test_split_load_kwargs_use_official_api_without_device(self):
        kw = model_load_kwargs([3.0, 4.0])
        self.assertNotIn('device', kw)          # engine asserts mutual exclusion
        self.assertEqual(kw['use_per_device'], [3.0, 4.0])
        self.assertEqual(kw['max_chunk_size'], MAX_CHUNK_SIZE)
        self.assertFalse(kw['progressbar'])

    def test_single_load_kwargs_keep_phase2_call(self):
        self.assertEqual(model_load_kwargs(None),
                         {'device': 'cuda:0', 'max_chunk_size': MAX_CHUNK_SIZE,
                          'progressbar': False})

    def test_boundary_sync_covers_all_active_devices_in_order(self):
        fake = FakeTorchForSync()
        sync = make_boundary_sync(fake, [0, 1])
        sync()
        self.assertEqual(fake.cuda.log, ['cuda:0', 'cuda:1'])
        fake.cuda.log.clear()
        make_boundary_sync(fake, [0])()          # single GPU: one identical call
        self.assertEqual(fake.cuda.log, ['cuda:0'])

    def test_cleanup_syncs_every_device_then_falls_back(self):
        fake = FakeTorchForSync()
        errors = []
        with patch.dict(sys.modules, {'exllamav3': None}):
            release_gpu_resources(fake, {}, errors, device_indices=[0, 1])
        self.assertEqual(fake.cuda.log,
                         ['empty_cache', 'cuda:0', 'cuda:1'])
        fake.cuda.log.clear()
        with patch.dict(sys.modules, {'exllamav3': None}):
            release_gpu_resources(fake, {}, errors)   # historical default-device call
        self.assertEqual(fake.cuda.log, ['empty_cache', None])


class RecordingObserver:
    """Duck-typed TransferObserver for the stage_window contract."""

    def __init__(self, log, fail_begin=False, fail_end=False):
        self.log = log
        self.snap = {'cuda_to_cuda': {'count': 1}}
        self.fail_begin, self.fail_end = fail_begin, fail_end
        self.armed = False

    def begin_window(self):
        if self.fail_begin:
            raise RuntimeError('synthetic begin failure')
        self.log.append('begin')
        self.armed = True

    def end_window(self):
        if self.fail_end:
            raise RuntimeError('synthetic end failure')
        self.log.append('end')
        self.armed = False
        return dict(self.snap)


class StageWindowObserverTests(unittest.TestCase):
    def base_sync(self, log):
        def sync():
            log.append('sync')
        return sync

    def run_window(self, fail_begin=False, fail_end=False):
        log = []
        obs = RecordingObserver(log, fail_begin=fail_begin, fail_end=fail_end)
        record = {}
        with stage_window(NoopControls(), 'stage=x', record, self.base_sync(log),
                          observer=obs):
            log.append('body')
        return record, log, obs

    def test_arm_after_opening_sync_snapshot_after_closing_sync(self):
        record, log, obs = self.run_window()
        self.assertEqual(log, ['sync', 'begin', 'body', 'sync', 'end'])
        # opening sync NOT observed; snapshot taken AFTER the closing sync
        self.assertEqual(record['transfers'], obs.snap)
        self.assertGreater(record['wall_s'], 0)
        self.assertFalse(obs.armed)                        # fully disarmed at exit

    def test_begin_failure_recorded_and_not_snapshotted(self):
        record, log, obs = self.run_window(fail_begin=True)
        self.assertTrue(any('transfer-observer-begin' in x for x in record['window_errors']))
        self.assertNotIn('transfers', record)
        self.assertNotIn('end', log)                       # end_window never attempted
        self.assertIn('body', log)                         # window still ran normally
        self.assertGreater(record['wall_s'], 0)

    def test_end_failure_recorded_window_still_closes(self):
        record, log, obs = self.run_window(fail_end=True)
        self.assertTrue(any('transfer-observer-end' in x for x in record['window_errors']))
        self.assertNotIn('transfers', record)
        self.assertGreater(record['wall_s'], 0)            # clock stopped before snapshot
        self.assertEqual(log[-1], 'sync')                  # end raised before appending

    def test_run_timed_job_snapshots_both_windows_only(self):
        # load/warm/gap excluded: exactly two arm/snapshot cycles per timed job,
        # and the observer log contains ONLY the window edges
        log = []
        obs = RecordingObserver(log)
        result, calls = self.exercise_job_with_observer(obs)
        self.assertEqual(result['problems'], [])
        self.assertEqual(result['prefill_window']['transfers'], obs.snap)
        self.assertEqual(result['decode_window']['transfers'], obs.snap)
        self.assertEqual(log, ['begin', 'end', 'begin', 'end'])
        self.assertEqual(len(calls), 4)                    # sync count unchanged by observer

    def exercise_job_with_observer(self, obs):
        sampler = types.ModuleType('exllamav3.generator.sampler')
        sampler.ArgmaxSampler = object
        calls = []

        def sync():
            calls.append(1)

        with patch.dict(sys.modules, {'exllamav3.generator.sampler': sampler}):
            result = run_timed_decode_job(
                FakeGenerator(), FakeJob, torch.zeros((1, 8192), dtype=torch.long),
                8192, 1, 1234, NoopControls(), 4, 3, sync, observer=obs)
        return result, calls


if __name__ == '__main__':
    unittest.main()
