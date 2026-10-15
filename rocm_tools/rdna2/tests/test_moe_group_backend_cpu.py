# SPDX-License-Identifier: MIT
"""An unavailable or incompatible optional bridge must retain the old route."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

path = Path(__file__).resolve().parents[3] / "exllamav3/rocm_py/moe_group_backend.py"
spec = importlib.util.spec_from_file_location("moe_backend_under_test", path)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class BackendTests(unittest.TestCase):
    def test_native_needs_no_bridge(self):
        native = SimpleNamespace(hgemm_exact_group=object())
        with patch.dict(sys.modules, {"exl3_moe_group_ext": None}):
            self.assertTrue(backend.ensure_group_backend(native))

    def test_missing_bridge(self):
        native = SimpleNamespace(__file__="original.so")
        with patch.dict(sys.modules, {"exl3_moe_group_ext": None}):
            self.assertFalse(backend.ensure_group_backend(native))
        self.assertFalse(hasattr(native, "hgemm_exact_group"))

    def test_incompatible_bridge(self):
        native = SimpleNamespace(__file__="original.so")
        bridge = SimpleNamespace(configure_engine=Mock(side_effect=RuntimeError("ABI")))
        with patch.dict(sys.modules, {"exl3_moe_group_ext": bridge}):
            self.assertFalse(backend.ensure_group_backend(native))
        self.assertFalse(hasattr(native, "hgemm_exact_group"))

    def test_reuse_configured_bridge(self):
        native = SimpleNamespace(__file__="original.so")
        bridge = SimpleNamespace(configure_engine=Mock(), hgemm_exact_group=object(),
                                 __file__="bridge.so")
        with patch.dict(sys.modules, {"exl3_moe_group_ext": bridge}):
            self.assertTrue(backend.ensure_group_backend(native))
            self.assertTrue(backend.ensure_group_backend(native))
        bridge.configure_engine.assert_called_once_with("original.so")
        self.assertIs(native.hgemm_exact_group, bridge.hgemm_exact_group)
        self.assertEqual(native._moe_exact_group_backend, "bridge.so")


if __name__ == "__main__":
    unittest.main()
