# SPDX-License-Identifier: MIT
"""Verify the default policy cannot expand beyond measured hardware/format."""

import importlib.util
import sys
import unittest
from pathlib import Path

path = Path(__file__).resolve().parents[3] / "exllamav3/rocm_py/moe_prefill_policy.py"
spec = importlib.util.spec_from_file_location("moe_policy_under_test", path)
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


class PolicyTests(unittest.TestCase):
    def config(self, **kwargs):
        result = {
            "arch": "gfx1030",
            "geometry": (2560, 640, 512, 256, 10, "silu", True),
            "projections": (
                (2560, 640, 3, False, True),
                (2560, 640, 3, False, True),
                (640, 2560, 3, False, True),
            ),
            "triton_compatible": True,
        }
        result.update(kwargs)
        return result

    def test_validated_pair(self):
        chosen = policy.select_policy(**self.config())
        self.assertTrue(chosen.enabled)
        self.assertEqual(chosen.fused_rows, 128)
        self.assertEqual(chosen.fusion, "fused")

    def test_optional_triton(self):
        chosen = policy.select_policy(**self.config(triton_compatible=False))
        self.assertFalse(chosen.enabled)
        self.assertEqual(chosen.fusion, "0")

    def test_arch_qualifiers(self):
        self.assertTrue(
            policy.select_policy(**self.config(arch="gfx1030:sramecc-:xnack-")).enabled
        )

    def test_other_hardware(self):
        for arch in ("gfx1201", "gfx1100", "gfx1031", "sm_90"):
            with self.subTest(arch=arch):
                self.assertFalse(policy.select_policy(**self.config(arch=arch)).enabled)

    def test_other_model_or_placement(self):
        original = self.config()["geometry"]
        for index, replacement in (
            (0, 4096),
            (1, 768),
            (2, 256),
            (3, 512),
            (4, 8),
            (5, "gelu"),
            (6, False),
        ):
            geometry = list(original)
            geometry[index] = replacement
            with self.subTest(geometry=geometry):
                self.assertFalse(
                    policy.select_policy(
                        **self.config(geometry=tuple(geometry))
                    ).enabled
                )

    def test_other_weight_format(self):
        for replacement in (
            (2560, 640, 4, False, True),
            (2560, 640, 3, True, False),
            (2560, 640, 3, False, False),
        ):
            projections = list(self.config()["projections"])
            projections[0] = replacement
            with self.subTest(projection=replacement):
                self.assertFalse(
                    policy.select_policy(
                        **self.config(projections=tuple(projections))
                    ).enabled
                )

    def test_numerical_contract(self):
        self.assertFalse(policy.select_policy(**self.config(folded=True)).enabled)
        self.assertFalse(
            policy.select_policy(**self.config(deterministic=False)).enabled
        )


if __name__ == "__main__":
    unittest.main()
