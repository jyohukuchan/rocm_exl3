"""Confidence robustness without importing the native inference extension."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    'draft_confidence_cpu', Path(__file__).resolve().parents[1] /
    'exllamav3/generator/draft_confidence.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Calibrator = module.DraftConfidenceCalibrator


class ConfidenceTests(unittest.TestCase):
    def test_nonfinite_scores_are_untrusted_during_and_after_burn_in(self):
        c = Calibrator(.6, burn_in=2, min_count=1)
        for score in (float('nan'), float('inf'), -float('inf')):
            self.assertEqual(c.estimate(score), 0.0)
        self.assertEqual(c.estimate(2.0), 1.0)
        c.add_label(2.0, True)
        c.add_label(2.0, False)
        self.assertEqual(c.estimate(2.0), .5)
        for score in (float('nan'), float('inf'), -float('inf')):
            self.assertEqual(c.estimate(score), 0.0)
        self.assertEqual(c.invalid_estimates, 6)

    def test_invalid_labels_do_not_change_learned_statistics(self):
        c = Calibrator(.6, burn_in=2, min_count=1)
        c.add_label(2.0, True)
        c.add_label(2.0, True)
        threshold = c.threshold()
        for score in (float('nan'), float('inf'), -float('inf')):
            c.add_label(score, False)
        self.assertEqual(c.total, 2)
        self.assertEqual(c.bins, {2: [2., 2.]})
        self.assertEqual(c.threshold(), threshold)
        self.assertEqual(c.estimate(2.0), 1.0)
        self.assertEqual(c.skipped_nonfinite_labels, 3)


if __name__ == '__main__':
    unittest.main()
