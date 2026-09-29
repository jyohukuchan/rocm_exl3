#!/usr/bin/env python3
"""CPU-only regression tests for GatedRMSNorm's tensor-parallel plumbing
(fix 148ae83: "preserve GDN sigmoid gating through TP export and import").

The Qwen3.8-Flash-Next TP2 smoke generated pure gibberish while layer-split on
the same pack was fine: the probe (q-layer-probe.json) showed the first block
diverging (relative L2 1.08 on both ranks, TP norms inflated ~1.6x) and the
cause was TP-specific, not quantization. GatedRMSNorm.tp_export omitted
``gate_activation`` from its kwargs, and tp_import / tp_import_split built the
BC_GatedRMSNorm without the trailing gate-activation enum, so every imported
norm silently took the constructor default ``"silu"``. Qwen4Exp's GDN output
norm is ``"sigmoid"`` (output_gate_type), so both ranks gated the linear-
attention output with silu(gate) instead of sigmoid(gate) and the residual
stream stack diverged from layer 0 onward.

The real exllamav3 package cannot be imported on a CPU-only box (pydantic / the
compiled extension), so gated_rmsnorm.py is loaded from source with its heavy
siblings faked, exactly like the other tests here fake their stack:

    real (from file):  util/misc, util/device_copy, util/tensor,
                       model/model_tp_alloc, modules/module,
                       modules/gated_rmsnorm
    faked:             exllamav3_ext (BC_GatedRMSNorm records its constructor
                       args; gated_rms_norm is a torch reference that applies
                       silu or sigmoid per the gate_act enum it receives, so a
                       lost enum changes the output), model/config,
                       the shared-memory producer/consumer (plain tensor tokens)

Proven properties:
  * tp_export ships gate_activation verbatim (silu AND sigmoid), and asserts
    on an export before load;
  * tp_import and tp_import_split reconstruct the activation, hand
    BC_GatedRMSNorm the same 6-argument form load() uses with the correct enum
    (sigmoid -> 1, silu -> 0), and produce non-grad weights with the exported
    weight values (1-D element-range and 2-D group-row split geometries);
  * a round-tripped module is forward-identical to its baseline in both
    activations, on the strict-fp32 sigmoid fallback (fp32 in) and through the
    ext reference on the bf16 hot path that the GDN output norm actually takes;
  * sensitivity: the pre-148ae83 implementations (reproduced verbatim here from
    b8f453b) are caught by every layer of the contract above - missing kwarg,
    default-silu reconstruction, 5-argument BC build, grad-requiring weights,
    and a forward that no longer matches the sigmoid baseline.

GPU-side behavior (collectives, the real kernels) is hardware-validated
separately; these tests pin the export/import plumbing only.

Run from the repo root (or anywhere):
    python3 -m pytest -q rocm_tools/rdna2/tests/test_tp_gated_norm_cpu.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch   # host CPU torch: no CUDA calls anywhere in this file

REPO = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# module-under-test loader: real source files, faked heavy siblings
# (same pattern as test_tp_qwen_arch_cpu.py, narrowed to gated_rmsnorm.py)
# ---------------------------------------------------------------------------

_SAVED = {}
_PLACED = {}
_INSTALLED = []


def _install(name, mod):
    _SAVED.setdefault(name, sys.modules.get(name))
    _PLACED[name] = mod
    sys.modules[name] = mod
    _INSTALLED.append(name)
    return mod


def _mk_pkg(name, path = None):
    mod = _install(name, types.ModuleType(name))
    mod.__path__ = path or []
    return mod


def _load_real(dotted, relpath):
    parent, _, leaf = dotted.rpartition(".")
    spec = importlib.util.spec_from_file_location(dotted, REPO / relpath)
    mod = _install(dotted, importlib.util.module_from_spec(spec))
    if parent:
        setattr(sys.modules[parent], leaf, mod)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# fakes: ext, config, producer/consumer
# ---------------------------------------------------------------------------

class FakeExt:
    """Stands in for exllamav3_ext.

    BC_GatedRMSNorm records every construction call (load() and the two TP
    import paths all build one); gated_rms_norm is a torch reference of the
    kernel that picks silu or sigmoid by the gate_act enum argument - so an
    import that loses the activation changes what forward() computes, exactly
    as the native kernel would.
    """

    def __init__(self):
        self.bc_calls = []

    class _BC:
        def __init__(self, ext, *args):
            ext.bc_calls.append(list(args))
            self.args = args

    def BC_GatedRMSNorm(self, *args):
        return self._BC(self, *args)

    def gated_rms_norm(self, x, weight, y, gate, eps, constant_bias,
                       groups, gate_first, gate_act):
        assert constant_bias == 0.0 and groups == 1, \
            "the CPU reference fake covers the simple per-channel case"
        gact = torch.sigmoid if gate_act else torch.nn.functional.silu
        h = x.float()
        if gate_first:
            h = h * gact(gate.float())
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim = True) + eps)
            h = weight.float() * h
        else:
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim = True) + eps)
            h = weight.float() * h
            h = h * gact(gate.float())
        y.copy_(h.to(y.dtype))


class FakeConfig:
    class _STC:
        def __init__(self, tensors):
            self.tensors = tensors

        def get_tensor(self, key, device = None, allow_bf16 = False, **kwargs):
            return self.tensors[key].clone()

    def __init__(self, tensors):
        self.stc = self._STC(tensors)
        self.infer_params = None


CPU = torch.device("cpu")
KEY = "layers.0.linear_attn.norm"
EPS = 1e-6


class FakeProducer:
    def send(self, tensor):
        return {"tensor": None if tensor is None else tensor.detach().clone()}


class FakeConsumer:
    def __init__(self, producer):
        self.producer = producer
        self.recvs = 0

    def recv(self, imp, cuda = False, **kwargs):
        self.recvs += 1
        return imp["tensor"].clone()


FAKE_EXT = FakeExt()


def _install_stack():
    global FAKE_EXT
    FAKE_EXT = FakeExt()
    _mk_pkg("_tp_gated_norm_cpu_harness", [str(REPO / "exllamav3")])
    ext_pkg = _mk_pkg("_tp_gated_norm_cpu_harness.ext")
    ext_pkg.exllamav3_ext = FAKE_EXT
    util = _mk_pkg("_tp_gated_norm_cpu_harness.util", [str(REPO / "exllamav3/util")])
    util.Timer = _load_real("_tp_gated_norm_cpu_harness.util.misc", "exllamav3/util/misc.py").Timer
    _load_real("_tp_gated_norm_cpu_harness.util.device_copy", "exllamav3/util/device_copy.py")
    _load_real("_tp_gated_norm_cpu_harness.util.tensor", "exllamav3/util/tensor.py")
    _mk_pkg("_tp_gated_norm_cpu_harness.model")
    cfg = _install("_tp_gated_norm_cpu_harness.model.config", types.ModuleType("_tp_gated_norm_cpu_harness.model.config"))
    cfg.Config = FakeConfig
    cfg.NullConfig = lambda: FakeConfig({})
    _load_real("_tp_gated_norm_cpu_harness.model.model_tp_alloc", "exllamav3/model/model_tp_alloc.py")
    mods = _mk_pkg("_tp_gated_norm_cpu_harness.modules", [str(REPO / "exllamav3/modules")])
    mods.Module = _load_real("_tp_gated_norm_cpu_harness.modules.module", "exllamav3/modules/module.py").Module
    _load_real("_tp_gated_norm_cpu_harness.modules.gated_rmsnorm", "exllamav3/modules/gated_rmsnorm.py")


def _uninstall_stack():
    # Identity-guarded: only drop names this file's stack still owns. Other CPU
    # stub suites install fakes under the same dotted names; if a later import
    # replaced ours, popping here would tear out THEIR stack mid-session (and the
    # dynamic `from ..model.config import NullConfig` in module.py would then
    # fall back to the real exllamav3 package on disk -> native extension load).
    for name in reversed(_INSTALLED):
        if sys.modules.get(name) is not _PLACED.get(name):
            continue
        old = _SAVED.get(name)
        if old is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = old
    _INSTALLED.clear()


_install_stack()

import torch.nn as _nn                                                        # noqa: E402
from _tp_gated_norm_cpu_harness.modules.gated_rmsnorm import GatedRMSNorm                      # noqa: E402

GRN = sys.modules["_tp_gated_norm_cpu_harness.modules.gated_rmsnorm"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def make_loaded(gate_activation, weight = None):
    """A GatedRMSNorm through the real load() path (which builds the BC with
    the correct enum), so the round-trip starts from a known-good baseline."""
    if weight is None:
        weight = torch.linspace(-0.5, 0.5, 16) + 1.0
    m = GatedRMSNorm(config = FakeConfig({f"{KEY}.weight": weight}), key = KEY,
                     rms_norm_eps = EPS, out_dtype = torch.float,
                     gate_activation = gate_activation)
    m.load(CPU)
    return m


def roundtrip(module, tp_import = None, split = None, export = None):
    """export -> (tp_)?import, with either side optionally swapped for the
    pre-fix implementation. Returns (imported, exported, bc_args) where
    bc_args is the BC_GatedRMSNorm constructor call recorded for the import."""
    prod = FakeProducer()
    exported = export(module, {}, prod) if export is not None \
        else module.tp_export(plan = {}, producer = prod)
    ctx = {"consumer": FakeConsumer(prod), "device": CPU}
    n_before = len(FAKE_EXT.bc_calls)
    target = tp_import if tp_import is not None else \
        (GatedRMSNorm.tp_import_split if split is not None else GatedRMSNorm.tp_import)
    with patch("torch.cuda.synchronize", lambda: None):
        imported = target(ctx, exported, {}, *([split] if split is not None else []))
    new_calls = FAKE_EXT.bc_calls[n_before:]
    assert len(new_calls) == 1, \
        f"expected exactly one BC_GatedRMSNorm build on import, got {len(new_calls)}"
    return imported, exported, new_calls[0]


def sample(x_dtype = torch.float):
    gen = torch.Generator().manual_seed(7)
    x = torch.randn((1, 4, 16), generator = gen).to(x_dtype)
    gate = torch.randn((1, 4, 16), generator = gen).float()
    return x, gate


# ---------------------------------------------------------------------------
# pre-148ae83 implementations, verbatim from b8f453b (the regression to catch)
# ---------------------------------------------------------------------------

def legacy_tp_export(self, plan, producer):
    assert self.device is not None, "Cannot export module for TP before loading."
    return {
        "cls": GatedRMSNorm,
        "kwargs": {
            "key": self.key,
            "rms_norm_eps": self.rms_norm_eps,
            "out_dtype": self.out_dtype,
            "constant_bias": self.constant_bias,
            "groups": self.groups,
            "gate_first": self.gate_first,
        },
        "weight": producer.send(self.weight),
        "device": self.device,
    }


def legacy_tp_import(local_context, exported, plan):
    consumer = local_context["consumer"]
    device = local_context["device"]
    module = GatedRMSNorm(
        config = None,
        **exported["kwargs"],
    )
    module.device = device
    w = consumer.recv(exported["weight"], cuda = True)
    module.weight = _nn.Parameter(w)
    module.bc = GRN.ext.BC_GatedRMSNorm(module.weight, module.rms_norm_eps,
                                        module.constant_bias, module.groups,
                                        module.gate_first)
    torch.cuda.synchronize()
    return module


def legacy_tp_import_split(local_context, exported, plan, split):
    consumer = local_context["consumer"]
    device = local_context["device"]
    first, last = split
    module = GatedRMSNorm(
        config = None,
        **exported["kwargs"],
    )
    module.device = device
    w = consumer.recv(exported["weight"], cuda = True)
    if w.dim() == 2:
        w = w[first : last, :]
    elif w.dim() == 1 and (last - first) < w.shape[0]:
        w = w[first : last]
    module.weight = _nn.Parameter(w.to(module.device).contiguous())
    module.bc = GRN.ext.BC_GatedRMSNorm(module.weight, module.rms_norm_eps,
                                        module.constant_bias, module.groups,
                                        module.gate_first)
    return module


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestGatedNormTpExport(unittest.TestCase):
    def setUp(self):
        FAKE_EXT.bc_calls.clear()

    # -- export metadata ----------------------------------------------------

    def test_export_carries_gate_activation(self):
        for act, enum in (("sigmoid", 1), ("silu", 0)):
            with self.subTest(gate_activation = act):
                base = make_loaded(act)
                exported = base.tp_export(plan = {}, producer = FakeProducer())
                self.assertIn("gate_activation", exported["kwargs"],
                              "tp_export must ship gate_activation (148ae83)")
                self.assertEqual(exported["kwargs"]["gate_activation"], act)
                # the rest of the constructor kwargs ride along unchanged
                for k, v in (("rms_norm_eps", EPS), ("out_dtype", torch.float),
                             ("constant_bias", 0.0), ("groups", 1),
                             ("gate_first", False)):
                    self.assertEqual(exported["kwargs"][k], v)

    def test_export_before_load_fails(self):
        m = GatedRMSNorm(config = None, key = KEY, rms_norm_eps = EPS)
        with self.assertRaises(AssertionError):
            m.tp_export(plan = {}, producer = FakeProducer())

    # -- import: activation + BC enum + weights ------------------------------

    def test_tp_import_preserves_activation_and_bc_enum(self):
        for act, enum in (("sigmoid", 1), ("silu", 0)):
            with self.subTest(gate_activation = act):
                base = make_loaded(act)
                # load() itself builds the BC with the 6-argument form + enum
                self.assertEqual(len(FAKE_EXT.bc_calls[-1]), 6)
                self.assertEqual(FAKE_EXT.bc_calls[-1][5], enum)
                imported, _, bc = roundtrip(base)
                self.assertEqual(imported.gate_activation, act,
                                 "imported module must not silently default to silu")
                self.assertEqual(len(bc), 6,
                                 "the TP import must build BC_GatedRMSNorm with the "
                                 "gate-activation enum, as load() does")
                self.assertEqual(bc[5], enum)
                self.assertTrue(torch.equal(imported.weight.data, base.weight.data))
                self.assertFalse(imported.weight.requires_grad)
                self.assertIs(imported.device, CPU)

    def test_tp_import_split_preserves_activation_and_bc_enum(self):
        w1d = torch.linspace(-1.0, 1.0, 32)
        w2d = torch.linspace(-1.0, 1.0, 32).view(2, 16)   # grouped (Mamba2-style) rows
        for act, enum in (("sigmoid", 1), ("silu", 0)):
            for weight, split in ((w1d, (8, 24)), (w2d, (1, 2))):
                with self.subTest(gate_activation = act, weight_dim = weight.dim()):
                    base = make_loaded(act, weight = weight.clone())
                    imported, _, bc = roundtrip(base, split = split)
                    self.assertEqual(imported.gate_activation, act)
                    self.assertEqual(len(bc), 6)
                    self.assertEqual(bc[5], enum)
                    expected = weight[slice(*split)] if weight.dim() == 1 \
                        else weight[split[0]: split[1], :]
                    self.assertEqual(tuple(imported.weight.shape), tuple(expected.shape))
                    self.assertTrue(torch.equal(imported.weight.data, expected))
                    self.assertFalse(imported.weight.requires_grad)

    # -- forward equivalence through the round trip ---------------------------

    def test_forward_roundtrip_bitexact(self):
        # sigmoid fp32 in: strict-fp32 torch fallback (no ext at all);
        # silu and bf16-sigmoid: the ext reference path, keyed by the enum the
        # imported BC/forward actually carries
        for act in ("sigmoid", "silu"):
            for dtype in (torch.float, torch.bfloat16):
                with self.subTest(gate_activation = act, dtype = dtype):
                    base = make_loaded(act)
                    imported, _, _ = roundtrip(base)
                    x, gate = sample(dtype)
                    y0 = base.forward(x, {}, gate = gate)
                    y1 = imported.forward(x, {}, gate = gate)
                    self.assertTrue(torch.equal(y0, y1),
                                    f"{act}/{dtype}: round-tripped norm is not "
                                    "forward-identical to its baseline")

    def test_forward_distinguishes_sigmoid_from_silu(self):
        # sanity for the equivalence check above: the two activations are far
        # apart on these inputs, so an enum lost in transit cannot pass it
        base_sig = make_loaded("sigmoid")
        base_silu = make_loaded("silu")
        x, gate = sample()
        y_sig = base_sig.forward(x, {}, gate = gate)
        y_silu = base_silu.forward(x, {}, gate = gate)
        self.assertFalse(torch.equal(y_sig, y_silu))

    # -- sensitivity: every layer of the contract catches the former omissions --

    def test_former_export_omission_is_caught(self):
        base = make_loaded("sigmoid")
        prod = FakeProducer()
        exported = legacy_tp_export(base, {}, prod)
        self.assertNotIn("gate_activation", exported["kwargs"],
                         "sanity: the pre-148ae83 export omitted the activation")

    def test_former_import_omissions_are_caught(self):
        base = make_loaded("sigmoid")
        # the full pre-fix pipeline: legacy export (no gate_activation kwarg)
        # into legacy import (constructor default silu, 5-argument BC, grad weight)
        imported, _, bc = roundtrip(base, tp_import = staticmethod(legacy_tp_import),
                                    export = legacy_tp_export)
        # 1) the enum is gone from the constructor call (5 args) ...
        self.assertEqual(len(bc), 5,
                         "sanity: the pre-148ae83 import built BC_GatedRMSNorm "
                         "without the gate-activation enum")
        # 2) ... the reconstructed module defaults to silu ...
        self.assertEqual(imported.gate_activation, "silu")
        # 3) ... its weight carries gradients ...
        self.assertTrue(imported.weight.requires_grad)
        # 4) ... and its forward no longer matches the sigmoid baseline
        x, gate = sample(torch.bfloat16)   # the bf16 hot path the GDN output norm takes
        y0 = base.forward(x, {}, gate = gate)
        y1 = imported.forward(x, {}, gate = gate)
        self.assertFalse(torch.equal(y0, y1),
                         "sanity: a silu-gated norm must not pass the equivalence test")

    def test_former_split_import_omissions_are_caught(self):
        base = make_loaded("sigmoid", weight = torch.linspace(-1.0, 1.0, 32))
        imported, _, bc = roundtrip(base, tp_import = staticmethod(legacy_tp_import_split),
                                    split = (8, 24), export = legacy_tp_export)
        self.assertEqual(len(bc), 5)
        self.assertEqual(imported.gate_activation, "silu")
        self.assertTrue(imported.weight.requires_grad)


def tearDownModule():
    _uninstall_stack()


if __name__ == "__main__":
    unittest.main(verbosity = 2)
