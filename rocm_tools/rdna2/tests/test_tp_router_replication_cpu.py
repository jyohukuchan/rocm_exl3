#!/usr/bin/env python3
"""CPU-only tests for the EXL3_TP_REPLICATE_ROUTER TP router-replication experiment
(exllamav3/modules/block_sparse_mlp.py).

Motivation (decode trace q-tp-ar-kv54-decode-probe-r2): every MoE layer broadcasts
selected_experts + routing_weights from the routing rank (96 collectives/forward on a
48-layer Qwen), and small RCCL messages are fixed-overhead-dominated. The experiment
computes the SAME global routing on every rank (std router only), so both broadcasts
disappear while local expert shards and the final reduction are untouched.

Proven here (plumbing only -- GPU numerics/speed are validated separately on hardware):
  * the opt-in defaults OFF and reads EXL3_TP_REPLICATE_ROUTER once on the parent;
  * tp_export ships the parent decision ("replicate_router"), tp_import follows the
    EXPORTED dict (never its own env), and older exported dicts without the key keep
    the legacy import shape;
  * unsupported router types (anything but "std") stay on the legacy path with an
    explicit printed reason;
  * allocation accounting: replicated router weight + the lazily built bsz-1 transpose
    move to storage_per_device (never token-multiplied), out of the split pool;
  * import ownership: replicated => gate on every rank (full global expert range),
    routing_device None, per-router metadata (per_expert_scale) received on every rank;
    zero-local-expert ranks included;
  * forward: the two routing broadcasts happen exactly on the legacy path and exactly
    zero on the replicated path; the trailing reduction/broadcast is untouched; both
    ranks compute identical global routing, and the summed expert output is identical
    to the legacy path's (and to a dense reference within fp tolerance).

The real exllamav3 package cannot be imported on a CPU-only box (pydantic / the
compiled extension), so block_sparse_mlp.py is loaded from source with its heavy
siblings faked, exactly like test_tp_gated_norm_cpu.py / test_tp_qwen_arch_cpu.py:

    real (from file):  util/misc, util/device_copy, util/tensor, model/model_tp_alloc,
                       modules/module, modules/block_sparse_mlp_cpu,
                       modules/block_sparse_mlp_routing, modules/block_sparse_mlp
    faked:             exllamav3_ext (routing_std + silu_mul are torch references; any
                       other ext attribute raises), model/config, tokenizer/mm_embedding,
                       multilinear, mlp, rmsnorm, layernorm, moe_batch_recon, the Linear
                       children (duck-typed, shared-memory producer/consumer as tensor
                       tokens, and the TP backend as a recording collective emulator)

Per-rank devices are torch.device("cpu:0") / ("cpu:1"): distinct identities, CPU
allocations. Harness modules register under a private per-process prefix, never under
the real "exllamav3" dotted name (same isolation contract as the sibling suites).

Run from the repo root (or anywhere):
    python3 -m pytest -q rocm_tools/rdna2/tests/test_tp_router_replication_cpu.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch            # noqa: E402  host CPU torch: no CUDA calls anywhere in this file
import torch.nn.functional as F   # noqa: E402

REPO = Path(__file__).resolve().parents[3]
_H = "_tp_router_repl_cpu_harness_" + str(os.getpid())
_INSTALLED = []


def _install(name, mod):
    assert name == _H or name.startswith(_H + "."), \
        f"harness module escapes the private namespace: {name}"
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
# fakes: ext, config, Linear / shared-expert children, producer/consumer
# ---------------------------------------------------------------------------

class FakeExt:
    """Stands in for exllamav3_ext. routing_std and silu_mul are torch references of
    the kernels (deterministic given the inputs, so rank-vs-rank equality is a real
    property of the plumbing, not of the fake). Anything else raises: the CPU paths
    under test must not touch the quantized fast paths."""

    def __init__(self):
        self.routing_calls = 0
        self.outputs = []   # (selected, weights) clones per routing_std call

    def silu_mul(self, g, u, a, limit):
        a.copy_((F.silu(g.float()) * u.float()).to(a.dtype))

    def routing_std(self, y, gate, router_logits, selected, weights,
                    per_expert_scale, gate_t, router_bias):
        self.routing_calls += 1
        logits = y.float() @ gate.float()
        if router_bias is not None:
            logits = logits + router_bias.float()
        router_logits.copy_(logits.to(router_logits.dtype))
        probs = torch.softmax(logits, dim = -1)
        tv, ti = probs.topk(weights.shape[-1], dim = -1)
        w = tv / tv.sum(dim = -1, keepdim = True)
        if per_expert_scale is not None:
            w = w * per_expert_scale.float()[ti]
        selected.copy_(ti.to(torch.long))
        weights.copy_(w.to(weights.dtype))
        self.outputs.append((selected.clone(), weights.clone()))

    def __getattr__(self, item):
        raise AssertionError(f"exllamav3_ext.{item} must not be called in CPU tests")


FAKE_EXT = FakeExt()


class FakeConfig:
    class _IP:
        no_reconstruct = False
    def __init__(self, stc = None):
        self.stc = stc
        self.infer_params = self._IP()


class _Inner:
    def __init__(self, weight):
        self.weight = weight
        self.bias = None


class FakeLinear:
    """Duck-typed loaded Linear. inner.weight is (in_features, out_features) (the
    routing gate is consumed directly as y @ gate, matching transposed_load)."""

    def __init__(self, config = None, key = None, in_features = 0, out_features = 0,
                 out_dtype = None, **kwargs):
        self.key = key
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype or torch.half
        self.quant_type = "fp16"
        self.device = None
        self.trim_padded_out = False
        self.out_features_unpadded = out_features
        seed = 17 + 31 * sum(ord(c) for c in (key or ""))
        gen = torch.Generator().manual_seed(seed)
        w = torch.randn(in_features, out_features, generator = gen) * 0.2
        self.inner = _Inner(w.half() if self.out_dtype != torch.float else w)

    def forward(self, x, params = None):
        return (x.float() @ self.inner.weight.float()).to(self.out_dtype)

    def storage_size(self):
        return 2 * self.in_features * self.out_features

    def recons_size(self):
        return 2 * self.in_features * self.out_features

    def optimizer_targets(self):
        return []

    def tp_export(self, plan, producer):
        assert self.device is not None, "export before load"
        return {
            "cls": FakeLinear,
            "kwargs": {"key": self.key, "in_features": self.in_features,
                       "out_features": self.out_features, "out_dtype": self.out_dtype},
            "weight": producer.send(self.inner.weight),
        }

    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        kw = exported["kwargs"]
        m = FakeLinear(key = kw["key"], in_features = kw["in_features"],
                       out_features = kw["out_features"], out_dtype = kw["out_dtype"])
        m.inner.weight = local_context["consumer"].recv(exported["weight"], cuda = True)
        m.device = local_context["device"]
        return m

    @staticmethod
    def tp_import_split(local_context, exported, plan, split, **kwargs):
        return FakeLinear.tp_import(local_context, exported, plan)


class FakeSharedMLP:
    """Duck-typed shared_experts child: a constant output, owned whole by one rank
    (empty plan range on the others, where it contributes exact zeros)."""

    def __init__(self, config = None, key = None, hidden = 0, const = None):
        self.key = key
        self.hidden = hidden
        self.const = const
        self.out_dtype = torch.float
        self.device = None

    def make_tp_allocation(self, options):
        return []

    def tp_export(self, plan, producer):
        return {
            "cls": FakeSharedMLP,
            "kwargs": {"key": self.key, "hidden": self.hidden},
            "const": producer.send(self.const),
        }

    @staticmethod
    def tp_import(local_context, exported, plan, **kwargs):
        kw = exported["kwargs"]
        m = FakeSharedMLP(key = kw["key"], hidden = kw["hidden"])
        first, last, _ = plan[kw["key"]]
        m.device = local_context["device"]
        m.const = local_context["consumer"].recv(exported["const"], cuda = True) \
            if last > first else None
        return m

    def forward(self, x, params = None):
        rows = x.reshape(-1, x.shape[-1]).shape[0]
        if self.const is None:
            return torch.zeros((rows, self.hidden), dtype = torch.float)
        return self.const.expand(rows, self.const.shape[-1]).clone()


class FakeProducer:
    def send(self, tensor):
        return {"tensor": None if tensor is None else tensor.detach().clone()}


class FakeConsumer:
    def __init__(self):
        self.recvs = []
    def recv(self, imp, cuda = False, **kwargs):
        self.recvs.append(imp)
        return None if imp is None or imp["tensor"] is None else imp["tensor"].clone()


def _install_stack():
    global FAKE_EXT
    FAKE_EXT = FakeExt()
    _mk_pkg(_H)
    ext_pkg = _mk_pkg(f"{_H}.ext")
    ext_pkg.exllamav3_ext = FAKE_EXT
    util = _mk_pkg(f"{_H}.util")
    util.Timer = _load_real(f"{_H}.util.misc", "exllamav3/util/misc.py").Timer
    _load_real(f"{_H}.util.device_copy", "exllamav3/util/device_copy.py")
    _load_real(f"{_H}.util.tensor", "exllamav3/util/tensor.py")
    util.profile_opt = None
    _mk_pkg(f"{_H}.model")
    cfg = _install(f"{_H}.model.config", types.ModuleType(f"{_H}.model.config"))
    cfg.Config = FakeConfig
    cfg.NullConfig = FakeConfig
    _load_real(f"{_H}.model.model_tp_alloc", "exllamav3/model/model_tp_alloc.py")
    _mk_pkg(f"{_H}.tokenizer")
    mme = _install(f"{_H}.tokenizer.mm_embedding",
                   types.ModuleType(f"{_H}.tokenizer.mm_embedding"))
    mme.FIRST_MM_EMBEDDING_INDEX = 1 << 40
    mods = _mk_pkg(f"{_H}.modules")
    mods.Module = _load_real(f"{_H}.modules.module", "exllamav3/modules/module.py").Module
    mods.Linear = FakeLinear
    for name, attrs in (("multilinear", {"MultiLinear": type("MultiLinear", (), {})}),
                        ("mlp", {"MLP": FakeSharedMLP, "GatedMLP": FakeSharedMLP}),
                        ("rmsnorm", {"RMSNorm": type("RMSNorm", (), {})}),
                        ("layernorm", {"LayerNorm": type("LayerNorm", (), {})}),
                        ("moe_batch_recon", {"PAD_MAX": 256})):
        m = _install(f"{_H}.modules.{name}", types.ModuleType(f"{_H}.modules.{name}"))
        for a, v in attrs.items():
            setattr(m, a, v)
    _load_real(f"{_H}.modules.block_sparse_mlp_cpu", "exllamav3/modules/block_sparse_mlp_cpu.py")
    _load_real(f"{_H}.modules.block_sparse_mlp_routing",
               "exllamav3/modules/block_sparse_mlp_routing.py")
    _load_real(f"{_H}.modules.block_sparse_mlp", "exllamav3/modules/block_sparse_mlp.py")
    for name in _INSTALLED:
        assert name == _H or name.startswith(_H + "."), name


def _uninstall_stack():
    # Identity-guarded (same contract as the sibling suites): never tear out another
    # file's stack if it replaced ours under the same dotted names.
    for name in reversed(_INSTALLED):
        if name in sys.modules:
            sys.modules.pop(name, None)
    _INSTALLED.clear()


_install_stack()

BSM = sys.modules[f"{_H}.modules.block_sparse_mlp"]
BlockSparseMLP = BSM.BlockSparseMLP
TPAllocator = sys.modules[f"{_H}.model.model_tp_alloc"].TPAllocator

# ---------------------------------------------------------------------------
# fakes: TP backend (a recording collective emulator with in-order data staging)
# ---------------------------------------------------------------------------

class FakeBackend:
    """Records every collective and moves broadcast data between the in-process rank
    modules through a shared arena keyed by each rank's collective slot counter. The
    counters only stay aligned if all ranks issue the SAME collective sequence -- which
    is exactly the invariant the routing-broadcast removal must not break (a mismatch
    fails here as a KeyError, mirroring a real deadlock)."""

    def __init__(self, rank, arena, log):
        self.rank = rank
        self.arena = arena
        self.log = log
        self.slot = 0

    def broadcast(self, t, src_device = None):
        self.log.append(("bcast", self.rank, src_device))
        key = f"bc{self.slot}"
        if src_device == self.rank:
            self.arena[key] = t.detach().clone()
        else:
            t.copy_(self.arena[key])
        self.slot += 1

    def all_reduce(self, t, contribution = True):
        self.log.append(("allreduce", self.rank, contribution))
        self.arena.setdefault(f"ar{self.slot}", []).append((self.rank, t.detach().clone()))
        self.slot += 1


# ---------------------------------------------------------------------------
# fixture: small std-router MoE, two ranks, expert-unit split
# ---------------------------------------------------------------------------

D0 = torch.device("cpu:0")
D1 = torch.device("cpu:1")
CPU = torch.device("cpu")

HIDDEN, INTER, E, K = 8, 4, 8, 2
MOE_KEY = "model.layers.0.mlp"
SHARED_KEY = f"{MOE_KEY}.shared_experts"
# gate row 0 (rest zero) x unit input -> logits [0,8,2,3,4,7,5,6]: top-2 picks {1, 5},
# one per rank under the (0,4)/(4,8) expert split
LOGITS_PATTERN = [0, 8, 2, 3, 4, 7, 5, 6]
# expert picks split across ranks; expected renormalized top-2 softmax weights
SEL_GLOBAL = torch.tensor([[1, 5]], dtype = torch.long)


def make_experts():
    gs, us, ds = [], [], []
    for e in range(E):
        gs.append(FakeLinear(key = f"{MOE_KEY}.experts.{e}.gate", in_features = HIDDEN,
                             out_features = INTER, out_dtype = torch.half))
        us.append(FakeLinear(key = f"{MOE_KEY}.experts.{e}.up", in_features = HIDDEN,
                             out_features = INTER, out_dtype = torch.half))
        ds.append(FakeLinear(key = f"{MOE_KEY}.experts.{e}.down", in_features = INTER,
                             out_features = HIDDEN, out_dtype = torch.float))
    for l in gs + us + ds:
        l.device = CPU
    return gs, us, ds


def make_parent(router_type = "std", shared = False, with_scale = False):
    gate = FakeLinear(key = f"{MOE_KEY}.gate", in_features = HIDDEN, out_features = E,
                      out_dtype = torch.half)
    w = torch.zeros(HIDDEN, E)
    w[0] = torch.tensor(LOGITS_PATTERN, dtype = torch.float)
    gate.inner.weight = w.half()
    gate.device = CPU
    gs, us, ds = make_experts()
    m = BlockSparseMLP(
        config = None, key = MOE_KEY, hidden_size = HIDDEN, intermediate_size = INTER,
        num_experts = E, num_experts_per_tok = K,
        router_type = router_type,
        routing_gate = gate, gates = gs, ups = us, downs = ds,
        shared_experts = FakeSharedMLP(key = SHARED_KEY, hidden = HIDDEN,
                                       const = torch.full((1, HIDDEN), 0.25)) if shared else None,
    )
    m.device = CPU
    if with_scale:
        m.per_expert_scale = torch.linspace(0.5, 1.2, E, dtype = torch.bfloat16)
    return m


def export_parent(parent):
    prod = FakeProducer()
    exported = parent.tp_export(plan = {}, producer = prod)
    return prod, exported


def make_plan(moe0 = (0, 4, "experts"), moe1 = (4, 8, "experts"), shared = None):
    plan = {D0: {MOE_KEY: moe0}, D1: {MOE_KEY: moe1}}
    if shared is not None:
        plan[D0][SHARED_KEY], plan[D1][SHARED_KEY] = shared
    return plan


def import_pair(exported, plan = None, output = D0):
    plan = plan if plan is not None else make_plan()
    mods, cons = {}, {}
    for d in (D0, D1):
        c = FakeConsumer()
        cons[d] = c
        ctx = {"consumer": c, "device": d, "output_device": output,
               "plan": plan, "active_devices": [D0, D1]}
        mods[d] = BlockSparseMLP.tp_import(ctx, exported, plan[d])
    return mods, cons


def run_forwards(mods, x = None):
    """bsz-1 decode forward on both ranks over one shared arena (rank order D0, D1)."""
    if x is None:
        x = torch.zeros((1, HIDDEN))
        x[0, 0] = 1.0
    arena, log = {}, []
    backends = {D0: FakeBackend(D0, arena, log), D1: FakeBackend(D1, arena, log)}
    outs = {}
    for d in (D0, D1):
        outs[d] = mods[d].forward(x, {"backend": backends[d]})
    return outs, arena, log


def captured_routing(m):
    """The routing tensors a rank actually used: its own bsz-1 CFG buffers when it owns
    a router, else the persistent broadcast targets."""
    if m.routing_gate is not None:
        return m.routing_cfg.selected_experts_bsz1.clone(), m.routing_cfg.routing_weights_bsz1.clone()
    return m.bcast_sel_bsz1.clone(), m.bcast_weights_bsz1.clone()


def arena_sum(arena, slot):
    parts = arena[f"ar{slot}"][0][1].clone()
    for _, p in arena[f"ar{slot}"][1:]:
        parts += p
    return parts


def count_bcasts(log, rank):
    return sum(1 for e in log if e[0] == "bcast" and e[1] == rank)


def count_allreduces(log, rank):
    return sum(1 for e in log if e[0] == "allreduce" and e[1] == rank)


def dense_reference(parent, bsz = 1):
    """Hand-computed routed sum (same op order as the fake routing kernel and the torch
    per-expert path), for the unit-row input used by run_forwards."""
    x = torch.zeros((bsz, HIDDEN))
    x[:, 0] = 1.0
    logits = x.float() @ parent.routing_gate.inner.weight.float()
    probs = torch.softmax(logits, dim = -1)
    tv, ti = probs.topk(K, dim = -1)
    w = (tv / tv.sum(dim = -1, keepdim = True)).half()
    out = torch.zeros(bsz, HIDDEN)
    for t in range(bsz):
        for j in range(K):
            e = int(ti[t, j])
            u = (x[t:t + 1].float() @ parent.ups[e].inner.weight.float()).half()
            g = (x[t:t + 1].float() @ parent.gates[e].inner.weight.float()).half()
            a = (F.silu(g.float()) * u.float()).half()
            d = (a.float() @ parent.downs[e].inner.weight.float()).float()
            out[t] += (d * w[t, j].float()).squeeze(0)
    return out


# ---------------------------------------------------------------------------
# tests: flag, parent decision, unsupported router types
# ---------------------------------------------------------------------------

class TestFlagAndExportDecision(unittest.TestCase):
    def setUp(self):
        BSM._TP_REPL_WARNED.clear()
        self.parent = make_parent()

    def test_default_off(self):
        # default OFF: the constant mirrors the env var, unset in the test env
        self.assertEqual(BSM.TP_REPLICATE_ROUTER,
                         os.environ.get("EXL3_TP_REPLICATE_ROUTER", "0") == "1")
        self.assertFalse(BSM.TP_REPLICATE_ROUTER)
        # experiment scoped to std routing only (Qwen router_type default)
        self.assertEqual(BSM.TP_REPLICATE_ROUTER_TYPES, ("std",))
        _, exported = export_parent(self.parent)
        self.assertFalse(exported["replicate_router"])

    def test_export_ships_parent_decision_and_keeps_old_keys(self):
        _, exported = export_parent(self.parent)
        legacy_keys = {"cls", "kwargs", "routing_gate", "shared_gate", "latent_in",
                       "latent_out", "e_score_correction_bias", "per_expert_scale",
                       "gates", "ups", "downs", "shared_experts", "device"}
        self.assertTrue(legacy_keys <= set(exported))
        self.assertEqual(exported["kwargs"]["router_type"], "std")
        self.assertTrue(exported["routing_gate"])   # the gate is already staged for all
        with patch.object(BSM, "TP_REPLICATE_ROUTER", True):
            _, exported = export_parent(self.parent)
            self.assertTrue(exported["replicate_router"])

    def test_unsupported_router_stays_legacy_with_reason(self):
        parent = make_parent(router_type = "ds3")
        buf = io.StringIO()
        with patch.object(BSM, "TP_REPLICATE_ROUTER", True):
            with contextlib.redirect_stdout(buf):
                _, exported = export_parent(parent)
        self.assertFalse(exported["replicate_router"])
        self.assertIn("EXL3_TP_REPLICATE_ROUTER", buf.getvalue())
        self.assertIn("legacy", buf.getvalue())
        # warning fires once per router type, not once per layer
        buf2 = io.StringIO()
        with patch.object(BSM, "TP_REPLICATE_ROUTER", True):
            with contextlib.redirect_stdout(buf2):
                parent.tp_router_replicated()
        self.assertEqual(buf2.getvalue(), "")
        # ... and the imported module keeps the full legacy broadcast plumbing
        mods, _ = import_pair(exported)
        self.assertIsNone(mods[D1].routing_gate)
        self.assertEqual(mods[D1].routing_device, D0)

    def test_gateless_module_never_replicates(self):
        self.parent.routing_gate = None
        with patch.object(BSM, "TP_REPLICATE_ROUTER", True):
            self.assertFalse(self.parent.tp_router_replicated())


# ---------------------------------------------------------------------------
# tests: TPAllocation accounting
# ---------------------------------------------------------------------------

class TestAllocation(unittest.TestCase):
    def setUp(self):
        BSM._TP_REPL_WARNED.clear()
        self.parent = make_parent()
        self.gate_bytes = self.parent.routing_gate.storage_size()
        self.expert_bytes = (sum(l.storage_size() for l in self.parent.gates)
                             + sum(l.storage_size() for l in self.parent.ups)
                             + sum(l.storage_size() for l in self.parent.downs))

    def test_legacy_keeps_gate_in_split_pool(self):
        with patch.object(BSM, "TP_REPLICATE_ROUTER", False):
            tpa, = self.parent.make_tp_allocation({})
        self.assertEqual(tpa.storage_to_split, self.gate_bytes + self.expert_bytes)
        self.assertEqual(tpa.storage_per_device, 0)

    def test_replicated_charges_gate_plus_lazy_transpose_per_device(self):
        with patch.object(BSM, "TP_REPLICATE_ROUTER", True):
            tpa, = self.parent.make_tp_allocation({})
        # gate + the lazily built RoutingCFG transpose, on every rank; the split pool
        # no longer carries the gate at all
        self.assertEqual(tpa.storage_per_device, 2 * self.gate_bytes)
        self.assertEqual(tpa.storage_to_split, self.expert_bytes)

    def test_replicated_storage_is_not_token_multiplied(self):
        with patch.object(BSM, "TP_REPLICATE_ROUTER", True):
            a1 = TPAllocator(self.parent.make_tp_allocation({}),
                             num_tokens = 1, output_num_tokens = 1)
            s1, store1, _ = a1.initial_split([10**12, 10**12])
            a2 = TPAllocator(self.parent.make_tp_allocation({}),
                             num_tokens = 4096, output_num_tokens = 4096)
            s2, store2, _ = a2.initial_split([10**12, 10**12])
        # per rank: replicated router bytes + half the expert pool (even split)
        expected = 2 * self.gate_bytes + self.expert_bytes // 2
        self.assertEqual(store1[0], expected)
        self.assertEqual(store1, store2,
                         "replicated router bytes must not scale with sequence length")


# ---------------------------------------------------------------------------
# tests: import ownership
# ---------------------------------------------------------------------------

class TestImportOwnership(unittest.TestCase):
    def setUp(self):
        BSM._TP_REPL_WARNED.clear()
        self.parent = make_parent(with_scale = True)

    def _export(self, replicate):
        with patch.object(BSM, "TP_REPLICATE_ROUTER", True):
            prod, exported = export_parent(self.parent)
        if not replicate:
            exported["replicate_router"] = False
        return exported

    def test_legacy_import_shape(self):
        mods, _ = import_pair(self._export(replicate = False))
        self.assertIsNotNone(mods[D0].routing_gate)
        self.assertIsNone(mods[D1].routing_gate)      # gate only on the output device
        self.assertEqual(mods[D0].routing_device, D0)
        self.assertEqual(mods[D1].routing_device, D0)
        self.assertIsNone(mods[D1].routing_cfg)

    def test_old_exported_dict_defaults_to_legacy(self):
        exported = self._export(replicate = True)
        del exported["replicate_router"]              # producer from before the experiment
        mods, _ = import_pair(exported)
        self.assertIsNone(mods[D1].routing_gate)
        self.assertEqual(mods[D1].routing_device, D0)

    def test_child_follows_exported_decision_not_its_own_env(self):
        with patch.object(BSM, "TP_REPLICATE_ROUTER", False):  # child env without the flag
            mods, _ = import_pair(self._export(replicate = True))
        self.assertIsNotNone(mods[D1].routing_gate)
        self.assertIsNone(mods[D0].routing_device)
        self.assertIsNone(mods[D1].routing_device)

    def test_replicated_imports_full_global_router_on_every_rank(self):
        mods, _ = import_pair(self._export(replicate = True))
        for d in (D0, D1):
            m = mods[d]
            self.assertIsNone(m.routing_device,
                              "replicated routers must not broadcast routing")
            self.assertIsNotNone(m.routing_cfg, "load_routing must run on every rank")
            # routing stays over ALL global experts, not the local shard
            self.assertEqual(m.routing_cfg.num_experts, E)
            self.assertEqual(m.routing_cfg.gate_tensor.shape, (HIDDEN, E))
            self.assertTrue(torch.equal(m.routing_cfg.gate_tensor,
                                         self.parent.routing_gate.inner.weight))
            # replicated per-router metadata (std routing reads per_expert_scale)
            self.assertIsNotNone(m.per_expert_scale)
            self.assertTrue(torch.equal(m.per_expert_scale, self.parent.per_expert_scale))
        # local expert shards untouched: different ranges, same global routing inputs
        self.assertEqual(mods[D0].routing_first, 0)
        self.assertEqual(mods[D1].routing_first, 4)

    def test_zero_local_expert_rank_replicated(self):
        plan = make_plan(moe0 = (0, E, "experts"), moe1 = (E, E, "experts"))
        mods, _ = import_pair(self._export(replicate = True), plan = plan)
        b = mods[D1]
        self.assertEqual(b.num_local_experts, 0)
        self.assertIsNone(b.experts_cfg)              # load_local skipped, as before
        self.assertIsNone(b.routing_gate)             # no routed work, no unbudgeted router
        self.assertIsNone(b.routing_cfg)
        self.assertIsNone(b.routing_device)
        self.assertEqual(b.tp_owner, D0)              # single owner -> trailing broadcast


# ---------------------------------------------------------------------------
# tests: forward collectives + routing results
# ---------------------------------------------------------------------------

class TestForwardCollectives(unittest.TestCase):
    def setUp(self):
        BSM._TP_REPL_WARNED.clear()
        self.parent = make_parent()

    def _mods(self, replicate, plan = None, parent = None):
        parent = parent if parent is not None else self.parent
        with patch.object(BSM, "TP_REPLICATE_ROUTER", True):
            prod, exported = export_parent(parent)
        if not replicate:
            exported["replicate_router"] = False
        mods, _ = import_pair(exported, plan = plan)
        return mods

    def test_bsz1_routing_broadcasts_removed_only_when_replicated(self):
        legacy_out, arena_l, log_l = run_forwards(self._mods(False))
        repl_out, arena_r, log_r = run_forwards(self._mods(True))
        for d in (D0, D1):
            self.assertEqual(count_bcasts(log_l, d), 2,
                             "legacy: selected_experts + routing_weights per layer")
            self.assertEqual(count_bcasts(log_r, d), 0,
                             "replicated: both routing broadcasts must be gone")
            self.assertEqual(count_allreduces(log_l, d), 1)
            self.assertEqual(count_allreduces(log_r, d),
                             count_allreduces(log_l, d),
                             "the final reduction must be untouched")

    def test_replicated_ranks_compute_identical_global_routing(self):
        mods = self._mods(True)
        BSM.ext.outputs.clear()
        run_forwards(mods)
        (sel_a, w_a), (sel_b, w_b) = BSM.ext.outputs
        self.assertTrue(torch.equal(sel_a, SEL_GLOBAL))
        self.assertTrue(torch.equal(sel_a, sel_b),
                        "both ranks must pick the same GLOBAL experts")
        self.assertTrue(torch.equal(w_a, w_b))
        # lazy transpose materialized on every rank now (the allocation's second copy)
        for d in (D0, D1):
            self.assertIsNotNone(mods[d].routing_cfg.gate_tensor_t)

    def test_legacy_broadcasts_deliver_owners_routing(self):
        mods = self._mods(False)
        run_forwards(mods)
        sel_a, w_a = captured_routing(mods[D0])
        sel_b, w_b = captured_routing(mods[D1])
        self.assertTrue(torch.equal(sel_a, sel_b))
        self.assertTrue(torch.equal(w_a, w_b))

    def test_rank_partials_and_final_sum_identical(self):
        _, arena_l, _ = run_forwards(self._mods(False))   # 2 bcasts before the reduce
        _, arena_r, _ = run_forwards(self._mods(True))    # reduce is slot 0
        self.assertTrue(torch.equal(arena_sum(arena_l, 2), arena_sum(arena_r, 0)),
                        "replicated routing must change the expert partials not at all")
        ref = dense_reference(self.parent)
        self.assertTrue(torch.allclose(arena_sum(arena_r, 0), ref, atol = 1e-5),
                        "summed local shards must reproduce the dense routed sum")

    def test_prefill_shape_routing_also_local(self):
        x = torch.zeros((4, HIDDEN))
        x[:, 0] = 1.0
        mods = self._mods(True)
        BSM.ext.outputs.clear()
        run_forwards(mods, x = x)
        (sel_a, _), (sel_b, _) = BSM.ext.outputs
        self.assertTrue(torch.equal(sel_a, sel_b))
        self.assertTrue(torch.equal(sel_a, SEL_GLOBAL.expand(4, K)))

    def test_single_owner_zero_expert_partner(self):
        plan = make_plan(moe0 = (0, E, "experts"), moe1 = (E, E, "experts"))
        mods = self._mods(True, plan = plan)
        outs, _, log = run_forwards(mods)
        for d in (D0, D1):
            self.assertEqual(count_bcasts(log, d), 1)   # only the owner's output broadcast
            self.assertEqual(count_allreduces(log, d), 0)
        self.assertTrue(torch.equal(outs[D0], outs[D1]))
        sel_a, _ = captured_routing(mods[D0])
        self.assertTrue(torch.equal(sel_a, SEL_GLOBAL))
        self.assertIsNone(mods[D1].routing_cfg)

    def test_shared_expert_output_preserved(self):
        parent = make_parent(shared = True)
        plan = make_plan(shared = ((0, 0, "channels"), (0, HIDDEN, "channels")))
        _, arena_l, _ = run_forwards(self._mods(False, plan = plan, parent = parent))
        _, arena_r, log_r = run_forwards(self._mods(True, plan = plan, parent = parent))
        self.assertEqual(count_bcasts(log_r, D0), 0)
        summed = arena_sum(arena_r, 0)
        ref = dense_reference(parent) + 0.25   # shared expert lives on the D1 shard
        self.assertTrue(torch.allclose(summed, ref, atol = 1e-5),
                        "routed sum + shared expert must survive the reduction")
        self.assertTrue(torch.equal(summed, arena_sum(arena_l, 2)))


def tearDownModule():
    _uninstall_stack()


if __name__ == "__main__":
    unittest.main(verbosity = 2)
