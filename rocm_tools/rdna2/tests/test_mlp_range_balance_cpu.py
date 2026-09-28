#!/usr/bin/env python3
"""CPU-only tests for the gfx1030 gated-MLP range-balance bookkeeping.

No GPU, no exllamav3 import (the package pulls in the compiled extension), no
mocked model: the helper's pair logic runs against tiny data-holder objects
and real fp16 CPU tensors, driving the actual code paths of
exllamav3/rocm_py/mlp_range_balance.py -- the part whose mistakes would
corrupt weights (marker/idempotency state machine, in-place-only mutation,
validate-before-touch, device and scope gates).

Run from the repo root (or anywhere):
    python3 -m unittest discover -s rocm_tools/rdna2/tests -v
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HELPER = REPO / "exllamav3" / "rocm_py" / "mlp_range_balance.py"

try:
    import torch
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False


def load_helper():
    """Load the module from its file without importing the exllamav3
    package (top level imports stdlib only, by design)."""
    spec = importlib.util.spec_from_file_location(
        "mrb_under_test", str(HELPER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if HAVE_TORCH:
    mrb = load_helper()


class Holders:
    """Plain data objects shaped like Linear/LinearEXL3 -- the attributes
    classify_pair/balance_pair actually read. No behaviour is mocked."""

    class Inner:
        def __init__(self, svh, bias = None):
            self.svh = svh
            self.bias = bias

    class Linear:
        def __init__(self, key, inner, quant_type = "exl3"):
            self.key = key
            self.inner = inner
            self.quant_type = quant_type
            self.lora_a_tensors = {}

    class Module:
        def __init__(self, up, down, act_limit = 0.0, key = "model.layers.0.mlp"):
            self.key = key
            self.ups = [up]
            self.downs = [down]
            self.act_limit = act_limit


def make_pair(u_vals = (0.5, 1.0, 2.0), d_vals = (1.0, 2.0, 4.0),
              u_quant = "exl3", d_quant = "exl3",
              u_bias = None, d_bias = None):
    svh_u = torch.tensor(u_vals, dtype = torch.half)
    svh_d = torch.tensor(d_vals, dtype = torch.half)
    up = Holders.Linear("mlp.up_proj", Holders.Inner(svh_u, u_bias))
    dn = Holders.Linear("mlp.down_proj", Holders.Inner(svh_d, d_bias))
    up.quant_type, dn.quant_type = u_quant, d_quant
    return up, dn, (svh_u.clone(), svh_d.clone())


@unittest.skipUnless(HAVE_TORCH, "torch not importable on this host")
class BalancePairTests(unittest.TestCase):

    def test_scales_reciprocally_once_and_marks(self):
        up, dn, _ = make_pair()
        status, _ = mrb.classify_pair(Holders.Module(up, dn), up, dn)
        self.assertEqual(status, "eligible")
        mrb.balance_pair(up, dn)
        self.assertTrue(torch.equal(up.inner.svh, torch.tensor([0.0625, 0.125, 0.25], dtype = torch.half)))
        self.assertTrue(torch.equal(dn.inner.svh, torch.tensor([8.0, 16.0, 32.0], dtype = torch.half)))
        self.assertEqual(getattr(up.inner, mrb.MARKER), 8.0)
        self.assertEqual(getattr(dn.inner, mrb.MARKER), 8.0)
        status, _ = mrb.classify_pair(Holders.Module(up, dn), up, dn)
        self.assertEqual(status, "already")

    def test_storage_pointers_survive_the_rescale(self):
        # BC_LinearEXL3 / MultiLinear pointer tables and CUDA graphs hold the
        # raw addresses: the rescale must be strictly in-place.
        up, dn, _ = make_pair()
        pu, pd = up.inner.svh.data_ptr(), dn.inner.svh.data_ptr()
        mrb.balance_pair(up, dn)
        self.assertEqual(up.inner.svh.data_ptr(), pu)
        self.assertEqual(dn.inner.svh.data_ptr(), pd)

    def test_reload_with_new_inner_balances_again_exactly_once(self):
        up, dn, orig = make_pair()
        mrb.balance_pair(up, dn)
        # Simulate a reload: load_exl3 rebuilds LinearEXL3 from the
        # checkpoint, so the marker is gone and svh holds fresh raw values.
        up.inner = Holders.Inner(orig[0].clone())
        dn.inner = Holders.Inner(orig[1].clone())
        status, _ = mrb.classify_pair(Holders.Module(up, dn), up, dn)
        self.assertEqual(status, "eligible")
        mrb.balance_pair(up, dn)
        # Not compounded: exactly /8 and *8 relative to the raw checkpoint.
        self.assertTrue(torch.equal(up.inner.svh, torch.tensor([0.0625, 0.125, 0.25], dtype = torch.half)))
        self.assertTrue(torch.equal(dn.inner.svh, torch.tensor([8.0, 16.0, 32.0], dtype = torch.half)))

    def test_mixed_markers_raise_and_change_nothing(self):
        up, dn, orig = make_pair()
        setattr(up.inner, mrb.MARKER, 8.0)          # one side from an old generation
        module = Holders.Module(up, dn)
        status, note = mrb.classify_pair(module, up, dn)
        self.assertEqual(status, "mixed")
        self.assertIn("up.inner", note)
        with self.assertRaises(RuntimeError):
            mrb.balance_module(module)
        self.assertTrue(torch.equal(dn.inner.svh, orig[1]))   # untouched

    def test_no_headroom_raises_before_touching(self):
        up, dn, orig = make_pair(d_vals = (8192.0, 10000.0))   # *8 overflows fp16
        with self.assertRaisesRegex(RuntimeError, "headroom"):
            mrb.balance_pair(up, dn)
        self.assertTrue(torch.equal(up.inner.svh, orig[0]))
        self.assertTrue(torch.equal(dn.inner.svh, orig[1]))
        self.assertIsNone(getattr(up.inner, mrb.MARKER, None))
        self.assertIsNone(getattr(dn.inner, mrb.MARKER, None))

    def test_nonfinite_source_raises_before_touching(self):
        up, dn, orig = make_pair()
        up.inner.svh[1] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            mrb.balance_pair(up, dn)
        self.assertTrue(torch.equal(dn.inner.svh, orig[1]))

    def test_underflow_raises_before_touching(self):
        up, dn, orig = make_pair(u_vals = (2.0 ** -24, -2.0 ** -24, 0.0))
        with self.assertRaisesRegex(RuntimeError, "underflow"):
            mrb.balance_pair(up, dn)
        self.assertTrue(torch.equal(up.inner.svh, orig[0]))
        self.assertTrue(torch.equal(dn.inner.svh, orig[1]))
        self.assertIsNone(getattr(up.inner, mrb.MARKER, None))
        self.assertIsNone(getattr(dn.inner, mrb.MARKER, None))

    def test_wrong_dtype_or_layout_refuses(self):
        up, dn, orig = make_pair()
        up.inner.svh = up.inner.svh.float()
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            mrb.balance_pair(up, dn)
        up.inner.svh = orig[0].clone()
        raw = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype = torch.half)
        view = raw[::2]                       # shape (3,), stride (2,)
        self.assertFalse(view.is_contiguous())
        dn.inner.svh = view
        before = view.clone()
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            mrb.balance_pair(up, dn)
        self.assertTrue(torch.equal(dn.inner.svh, before))
        self.assertTrue(torch.equal(up.inner.svh, orig[0]))

    def test_ineligible_pairs_are_skipped_not_marked(self):
        module_kwargs = dict(act_limit = 0.0)
        cases = {
            "fp16 up": dict(u_quant = "fp16"),
            "fp16 down": dict(d_quant = "fp16"),
            "up bias": dict(u_bias = torch.zeros(3, dtype = torch.half)),
            "down bias": dict(d_bias = torch.zeros(3, dtype = torch.half)),
        }
        for name, kw in cases.items():
            up, dn, orig = make_pair(**kw)
            module = Holders.Module(up, dn, **module_kwargs)
            status, _ = mrb.classify_pair(module, up, dn)
            self.assertEqual(status, "skip", f"{name} must skip")
            mrb.balance_module(module)                 # must be a no-op
            self.assertTrue(torch.equal(up.inner.svh, orig[0]), name)
            self.assertTrue(torch.equal(dn.inner.svh, orig[1]), name)
            self.assertIsNone(getattr(up.inner, mrb.MARKER, None), name)

    def test_act_limit_skips_lora_and_partial_override_conflict(self):
        up, dn, orig = make_pair()
        self.assertEqual(mrb.classify_pair(Holders.Module(up, dn, act_limit = 7.0), up, dn)[0], "skip")
        # LoRA delta on either side: conflict, declined while unbalanced.
        up.lora_a_tensors = {"l": object()}
        module = Holders.Module(up, dn)
        self.assertEqual(mrb.classify_pair(module, up, dn)[0], "conflict")
        mrb.balance_module(module)
        self.assertTrue(torch.equal(up.inner.svh, orig[0]))
        self.assertTrue(torch.equal(dn.inner.svh, orig[1]))
        self.assertIsNone(getattr(up.inner, mrb.MARKER, None))

    def test_override_state(self):
        up, dn, _ = make_pair()
        class Ovr:
            def __init__(self, inner):
                self.inner = inner
        self.assertEqual(mrb.override_state({}, up, dn), "none")
        self.assertEqual(mrb.override_state(None, up, dn), "none")
        self.assertEqual(mrb.override_state({"ovr": {}}, up, dn), "none")
        # override registered but routing to the same inner -> "none"
        self.assertEqual(mrb.override_state({"ovr": {up.key: Ovr(up.inner), dn.key: Ovr(dn.inner)}}, up, dn), "none")
        self.assertEqual(mrb.override_state({"ovr": {up.key: Ovr(None)}}, up, dn), "partial")
        self.assertEqual(mrb.override_state({"ovr": {dn.key: Ovr(None)}}, up, dn), "partial")
        self.assertEqual(mrb.override_state({"ovr": {up.key: Ovr(None), dn.key: Ovr(None)}}, up, dn), "full")

    def test_partial_override_declines_when_unbalanced(self):
        up, dn, orig = make_pair()
        class Ovr:
            def __init__(self, inner):
                self.inner = inner
        params = {"ovr": {up.key: Ovr(None)}}
        module = Holders.Module(up, dn)
        self.assertEqual(mrb.classify_pair(module, up, dn, params)[0], "conflict")
        mrb.balance_module(module, params)
        self.assertTrue(torch.equal(up.inner.svh, orig[0]))
        self.assertTrue(torch.equal(dn.inner.svh, orig[1]))

    def test_full_override_balances_base_inertly(self):
        # Both sides overridden: base svh is not read while active, so the
        # later un-overridden forwards still need the balance -> eligible.
        up, dn, orig = make_pair()
        class Ovr:
            def __init__(self, inner):
                self.inner = inner
        params = {"ovr": {up.key: Ovr(None), dn.key: Ovr(None)}}
        module = Holders.Module(up, dn)
        self.assertEqual(mrb.classify_pair(module, up, dn, params)[0], "eligible")
        mrb.balance_module(module, params)
        self.assertTrue(torch.equal(up.inner.svh, orig[0] / 8))
        self.assertTrue(torch.equal(dn.inner.svh, orig[1] * 8))

    def test_balanced_pair_with_late_conflict_raises(self):
        # Once marked, a late LoRA delta or a partial override must NOT be
        # silently balanced against the other side -- fail clearly instead.
        up, dn, orig = make_pair()
        module = Holders.Module(up, dn)
        mrb.balance_module(module)                     # marked
        dn.lora_a_tensors = {"l": object()}
        with self.assertRaisesRegex(RuntimeError, "already balanced"):
            mrb.balance_module(module)
        dn.lora_a_tensors = {}
        class Ovr:
            def __init__(self, inner):
                self.inner = inner
        with self.assertRaisesRegex(RuntimeError, "already balanced"):
            mrb.balance_module(module, {"ovr": {up.key: Ovr(None)}})
        # still single-balanced (not compounded) by the raise itself
        self.assertTrue(torch.equal(up.inner.svh, orig[0] / 8))

    def test_unloaded_pair_skips(self):
        up, dn, _ = make_pair()
        up.inner = None
        self.assertEqual(mrb.classify_pair(Holders.Module(up, dn), up, dn)[0], "skip")


@unittest.skipUnless(HAVE_TORCH, "torch not importable on this host")
class GuardAndWrapTests(unittest.TestCase):
    """The device gate + install path, as far as they can be exercised
    honestly on a CPU host."""

    def test_maybe_balance_is_a_noop_off_hip(self):
        # A non-HIP host must not even look at the pairs: is_hip_build() is
        # False here (torch.version.hip is None), so the guard short-circuits.
        self.assertFalse(mrb.is_hip_build())
        up, dn, orig = make_pair()
        module = Holders.Module(up, dn)
        x = torch.zeros((1, 4, 3), dtype = torch.half)
        mrb._maybe_balance(module, x, {})
        self.assertTrue(torch.equal(up.inner.svh, orig[0]))
        self.assertTrue(torch.equal(dn.inner.svh, orig[1]))

    def test_device_gate_rejects_other_devices(self):
        self.assertFalse(mrb.device_is_gfx1030(torch.device("cpu")))
        self.assertFalse(mrb.device_is_gfx1030("cuda:0"))   # no CUDA runtime here
        self.assertTrue(mrb.arch_supported("gfx1030"))
        self.assertTrue(mrb.arch_supported("gfx1030:some-suffix"))
        self.assertFalse(mrb.arch_supported("gfx1100"))
        self.assertFalse(mrb.arch_supported("gfx1031"))      # unmeasured sibling
        self.assertFalse(mrb.arch_supported(""))

    def test_wrap_forward_is_idempotent_and_preserves_the_original(self):
        calls = []

        class FakeMLP:
            def forward(self, x, params, out_dtype = None):
                calls.append(1)
                return "sentinel"

        note = mrb.wrap_forward(FakeMLP)
        self.assertIn("guarded", note)
        self.assertTrue(getattr(FakeMLP.forward, mrb.WRAPPER, False))
        note2 = mrb.wrap_forward(FakeMLP)               # second install
        self.assertIn("already", note2)
        inst = FakeMLP()
        out = FakeMLP.forward(inst, torch.zeros(1), {})  # guard runs, then math
        self.assertEqual(out, "sentinel")
        self.assertEqual(len(calls), 1)                  # exactly one dispatch


class HelperFileTests(unittest.TestCase):
    """Checks that hold even without torch."""

    def test_helper_top_level_is_stdlib_only(self):
        src = HELPER.read_text(encoding = "utf-8")
        for line in src.splitlines():
            if line.startswith(("import ", "from ")) and "future" not in line:
                self.fail(f"top-level import outside stdlib: {line!r}")

    def test_helper_is_loadable_standalone(self):
        mod = load_helper()                             # no exllamav3 package init
        self.assertEqual(mod.SCALE, 8.0)
        self.assertTrue(mod.MARKER.startswith("_rocm_py"))

    def test_rocm_py_wires_the_switch(self):
        src = (REPO / "exllamav3" / "rocm_py" / "__init__.py").read_text(encoding = "utf-8")
        self.assertIn("EXL3_ROCM_MLP_RANGE_BALANCE", src)
        self.assertIn("mlp_range_balance", src)


if __name__ == "__main__":
    unittest.main()
