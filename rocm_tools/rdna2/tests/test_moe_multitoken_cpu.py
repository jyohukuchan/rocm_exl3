#!/usr/bin/env python3
"""CPU-only contract tests for the EXL3_ROCM_MOE_MULTI_TOKEN route.

No GPU, no exllamav3 import (the package pulls in the compiled extension), no
source-string assertions: exllamav3/rocm_py/__init__.py is loaded standalone
(its module level is stdlib-only) and driven against real CPU tensors through
a FakeExt that reproduces the barrier-free mgemv launch contract read from
exllamav3_ext/rocm/quant/exl3_mgemv_rdna.hip:

  - bszm_in/bszm_out capped at the indices width; A is broadcast from row 0
    only when bszm_in == 1, addressed PER SLOT otherwise
  - min_index >= 0 with num_tokens == 1: in-range picks COMPACT into the
    leading slots (weights follow the original position)
  - min_index < 0: indices address local matrices directly; a -1 index skips
    its slot -- the dot/rotate stages never touch that slot's C row
  - the weighted grouped reduce sums stride = bszm / num_tokens CONTIGUOUS
    slot rows into row t UNCONDITIONALLY: no -1 guard (the cooperative kernel
    guards, the fast path does not), so the caller must have zeroed inactive
    rows before the call
  - under the fused epilogue (EXL3_GEMV_FUSE_OUT != "0") a masked weighted
    call's arrival counter never completes, the reduce NEVER RUNS, and the
    dirty counters poison every LATER weighted call too

Stages resolve from the pointer-table argument (like the real kernel's B
list), not from output-buffer identity: the candidate may write into
freshly-allocated module-owned scratch bundles, whose addresses the caching
allocator can recycle.

Whether the zeroed-slot sum reproduces the cooperative guarded result on
REAL hardware is for rocm_tools/rdna2/moe_multitoken_probe.py to measure;
nothing here claims it.

Run from the repo root (or anywhere):
    PYTHONPYCACHEPREFIX=/tmp/decode-opt2-pycache python3 -m pytest -q \
        -p no:cacheprovider rocm_tools/rdna2/tests/test_moe_multitoken_cpu.py
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ROCM_PY = REPO / "exllamav3" / "rocm_py" / "__init__.py"
FUSE_ENV = "EXL3_GEMV_FUSE_OUT"
MGEMV_ENV = "EXL3_MGEMV"

try:
    import torch
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False


def load_rocm_py(name="moe_mt_under_test"):
    """Load the file standalone: its module level is __future__ + os/threading."""
    spec = importlib.util.spec_from_file_location(name, str(ROCM_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if HAVE_TORCH:
    rp = load_rocm_py()


class FakeCache:
    """g_tensor_cache is still a wiring precondition of the route gate;
    the candidate itself keeps staging on the module (aliasing same-shaped
    layers/target/draft through a shape-keyed global cache is exactly what
    the design forbids), so this fake only has to exist and stay empty-able."""

    def __init__(self):
        self.cache = {}

    def get(self, device, shape, dtype, name=""):
        key = (str(device), tuple(shape), str(dtype), name)
        if key not in self.cache:
            self.cache[key] = torch.empty(shape, dtype=dtype)
        return self.cache[key]


class FakeExt:
    """CPU stand-in for ext.exl3_mgemm implementing the contract above.

    register(mod) builds one set of per-stage expert tables per module,
    sized by the module's LOCAL pointer-table width, and maps each stage's
    ptrs_trellis address to (stage, tables). A rebased index beyond the
    table width raises -- a rebase into an un-sliced full-expert table
    cannot pass silently.
    """

    def __init__(self):
        self.stage_of = {}
        self.calls = []
        self.poisoned = False
        self.hazards = 0
        self.raise_on_stage = None

    def register(self, mod, seed=7):
        cfg = mod.experts_cfg
        Hi = cfg.yh.shape[-1]
        I = cfg.interm_g.shape[-1]
        Ho = cfg.out_d.shape[-1]
        width = int(mod.multi_up.ptrs_trellis.shape[0])
        g = torch.Generator().manual_seed(seed + 1000 * width)
        tables = {
            "gate": [torch.randn(Hi, I, generator=g) * 0.1 for _ in range(width)],
            "up":   [torch.randn(Hi, I, generator=g) * 0.1 for _ in range(width)],
            "down": [torch.randn(I, Ho, generator=g) * 0.1 for _ in range(width)],
        }
        for ml, st in ((mod.multi_gate, "gate"), (mod.multi_up, "up"),
                       (mod.multi_down, "down")):
            if ml is not None:
                self.stage_of[ml.ptrs_trellis.data_ptr()] = (st, tables)
        return tables

    @staticmethod
    def _resolve(idx_vals, bszm, min_index, max_index, num_tokens):
        """[(slot, weight_position, local_mat)] + packed count."""
        if min_index >= 0:
            if num_tokens > 1:
                raise AssertionError(
                    "the fast path declines num_tokens>1 with min_index>=0; "
                    "the candidate must pass -1/-1 with pre-rebased indices")
            seq = [(v - min_index, pos) for pos, v in enumerate(idx_vals)
                   if min_index <= v < max_index]
            return [(slot, pos, mat) for slot, (mat, pos) in enumerate(seq)], len(seq)
        return [(j, j, idx_vals[j]) for j in range(bszm)], bszm

    def exl3_mgemm(self, A, B, C, suh, A_had, svh, indices, weights,
                   K, force_shape_idx, mcg, mul1, min_index, max_index,
                   force_num_sms, num_tokens, size_n_list=None, c_ptrs=None):
        assert A.dim() == 3 and C.dim() == 3, "mgemm wants (bszm,m,k)/(bszm,m,n)"
        assert indices.dim() == 2, "indices (1,bszm)"
        assert A.dtype == torch.half, "A must be half"
        assert not (mcg and mul1), "both codebook flags"
        assert A_had.data_ptr() != A.data_ptr(), "A_had must not alias A"
        stage, tables = self.stage_of[B.data_ptr()]
        ni = indices.size(1)
        bszm_in = min(A.size(0), ni)
        bszm_out = min(C.size(0), ni)
        bszm = max(bszm_in, bszm_out)
        idx_vals = [int(v) for v in indices.reshape(-1)][:bszm]
        rec = {"stage": stage, "min_index": min_index, "max_index": max_index,
               "num_tokens": num_tokens, "idx": idx_vals, "A": A.detach().clone(),
               "weights": None if weights is None else weights.reshape(-1).clone(),
               "env_fuse": os.environ.get(FUSE_ENV)}

        if weights is not None and min_index < 0:
            inactive = [j for j in range(bszm) if idx_vals[j] < 0]
            if inactive:
                rows = C[inactive, 0, :].float()
                rec["inactive_c_absmax_at_entry"] = float(rows.abs().max())

        if self.raise_on_stage == stage:
            self.calls.append(rec)
            raise RuntimeError(f"fake ext boom at {stage}")

        resolved, packed = self._resolve(idx_vals, bszm, min_index, max_index,
                                          num_tokens)
        for slot, wpos, mat in resolved:
            if mat < 0:
                continue        # masked: dot/rotate skip it, its C row is untouched
            assert 0 <= mat < len(tables[stage]), (
                f"local id {mat} outside table width {len(tables[stage])}: the "
                "pointer table was not sliced to the shard range")
            a = A[0, 0] if bszm_in == 1 else A[slot, 0]
            v = a.float() @ tables[stage][mat]
            if weights is not None:
                v = v * float(weights.reshape(-1)[wpos])
            C[slot, 0, :v.numel()] = v.to(C.dtype)
        self.calls.append(rec)

        if weights is None:
            return
        masked = min_index < 0 and any(v < 0 for v in idx_vals)
        fused = os.environ.get(FUSE_ENV, "1") != "0"
        if fused and ((masked and num_tokens > 1)
                      or (min_index >= 0 and packed == 0)):
            # arrivals < red_target: the reduce never runs NOW and the
            # counters stay dirty for every later weighted call as well
            self.hazards += 1
            self.poisoned = True
            return
        if self.poisoned:
            return
        stride = packed if min_index >= 0 else bszm // num_tokens
        for t in range(num_tokens):
            base = 0 if min_index >= 0 else t * stride
            acc = torch.zeros(C.size(2), dtype=torch.float)
            for jj in range(stride):
                acc = acc + C[base + jj, 0, :].float()
            C[t, 0, :] = acc.to(C.dtype)


class _ML:
    def __init__(self, width, K=4):
        self.ptrs_trellis = torch.arange(width, dtype=torch.long)
        self.ptrs_suh = torch.arange(width, dtype=torch.long)
        self.ptrs_svh = torch.arange(width, dtype=torch.long)
        self.K = K
        self.mcg = False
        self.mul1 = False


class _Cfg:
    def __init__(self, r_cap, K, Hi, I, Ho, mine, maxe):
        rows = r_cap * K
        self.yh = torch.zeros(rows, 1, Hi, dtype=torch.half)
        self.interm_g = torch.zeros(rows, 1, I, dtype=torch.half)
        self.interm_u = torch.zeros(rows, 1, I, dtype=torch.half)
        self.interm_a = torch.zeros(rows, 1, I, dtype=torch.half)
        self.out_d = torch.zeros(rows, 1, Ho, dtype=torch.float)
        self.out_bszn = torch.zeros(r_cap, Ho, dtype=torch.float)
        self.min_expert = mine
        self.max_expert = maxe
        self.out_d.normal_(0, 0.02)         # stale scratch, like torch.empty
        self.out_bszn.normal_(0, 0.02)


class _Mod:
    def __init__(self, E_local, r_cap=8, K=4, mine=-1, maxe=-1, gated=True,
                 Hi=128, I=128, Ho=128, act_limit=0.0, Kq=4):
        self.gated = gated
        self.act_limit = act_limit
        self.experts_cfg = _Cfg(r_cap, K, Hi, I, Ho, mine, maxe)
        self.multi_gate = _ML(E_local, Kq) if gated else None
        self.multi_up = _ML(E_local, Kq)
        self.multi_down = _ML(E_local, Kq)
        self.act_calls = []

    def activation_fn_call(self, g, u, a, limit):
        self.act_calls.append(limit)
        a.copy_(torch.nn.functional.silu(g.float()) * u.float())


def expected_rows(tables, y, sel, rw, lo, hi):
    """Independent expectation: per token, sum of w * (silu(y@Wg)*(y@Wu)@Wd)
    over the IN-RANGE picks with local index sel - lo (masked picks
    contribute nothing)."""
    r, K = sel.shape
    out = []
    for t in range(r):
        acc = torch.zeros(y.shape[1], dtype=torch.float)
        for k in range(K):
            v = int(sel[t, k])
            if lo is not None and not (lo <= v < hi):
                continue
            mat = v if lo is None else v - lo
            g = y[t].float() @ tables["gate"][mat]
            u = y[t].float() @ tables["up"][mat]
            a = torch.nn.functional.silu(g) * u
            acc = acc + (a @ tables["down"][mat]) * float(rw[t, k])
        out.append(acc)
    return torch.stack(out)


if HAVE_TORCH:

    class MTBase(unittest.TestCase):
        def setUp(self):
            self.saved = dict(rp._MOE_MT)
            self.saved_env = (os.environ.get(FUSE_ENV), os.environ.get(MGEMV_ENV))
            self.ext = FakeExt()
            self.tcache = FakeCache()
            rp._MOE_MT.update({
                "on": True, "reason": "test",
                "torch": torch, "ext": self.ext, "tcache": self.tcache,
                "arch_ok": lambda dev: True, "arch_cache": {},
                "require_cuda": False})

        def tearDown(self):
            rp._MOE_MT.clear()
            rp._MOE_MT.update(self.saved)
            for nm, val in zip((FUSE_ENV, MGEMV_ENV), self.saved_env):
                if val is None:
                    os.environ.pop(nm, None)
                else:
                    os.environ[nm] = val

        def make(self, r, k=4, E=8, mine=-1, maxe=-1, sel_rows=None, **kw):
            """Module + registered tables + (y, sel, rw). Shard modules get
            LOCAL pointer tables of width maxe-mine, like a TP rank's
            MultiLinear (rebase-into-an-unsliced-table cannot happen)."""
            width = (maxe - mine) if mine >= 0 else E
            m = _Mod(width, K=k, mine=mine, maxe=maxe, **kw)
            tables = self.ext.register(m)
            g = torch.Generator().manual_seed(11 + r * 100 + (mine + 1) * 7)
            y = (torch.randn(r, m.experts_cfg.yh.shape[-1], generator=g) * 0.25).half()
            if sel_rows is not None:
                sel = torch.tensor(sel_rows, dtype=torch.long)
            else:
                sel = torch.stack([torch.randperm(E, generator=g)[:k]
                                   for _ in range(r)]).long()
            rw = torch.softmax(torch.randn(r, k, generator=g), dim=-1).half()
            return m, y.contiguous(), sel.contiguous(), rw.contiguous(), tables


    class TestDefaultsAndWiring(MTBase):

        def test_fresh_module_is_off_and_unwired(self):
            fresh = load_rocm_py("moe_mt_fresh")
            st = fresh.moe_multi_token_status()
            self.assertFalse(st["on"])
            self.assertIn("EXL3_ROCM_MOE_MULTI_TOKEN", st["reason"])
            m, y, sel, rw, _t = self.make(3)
            self.assertFalse(fresh.moe_multi_token_step(m, y, sel, rw))
            ok, why = fresh.moe_multi_token_supported(m, y, sel, rw)
            self.assertFalse(ok)
            self.assertIn("not wired", why)
            self.assertEqual(self.ext.calls, [])

        def test_on_flag_false_never_calls_candidate(self):
            rp._MOE_MT["on"] = False
            m, y, sel, rw, _t = self.make(3)
            self.assertFalse(rp.moe_multi_token_step(m, y, sel, rw))
            self.assertEqual(self.ext.calls, [])
            rp.moe_mgemm_bszN(m, y, sel, rw)              # dispatch -> row loop
            self.assertEqual(len(self.ext.calls), 9)      # 3 rows x (gate, up, down)
            self.assertTrue(all(c["num_tokens"] == 1 for c in self.ext.calls))

        def test_mgemv_disabled_falls_back_to_rowloop(self):
            # required guard: with the native fast path switched off the
            # candidate must decline and the unchanged row loop must own
            # every batch size, including rows 2..5
            os.environ[MGEMV_ENV] = "0"
            m, y, sel, rw, tables = self.make(3)
            ok, why = rp.moe_multi_token_supported(m, y, sel, rw)
            self.assertFalse(ok)
            self.assertIn("EXL3_MGEMV", why)
            self.assertFalse(rp.moe_multi_token_step(m, y, sel, rw))
            self.assertEqual(self.ext.calls, [])
            rp.moe_mgemm_bszN(m, y, sel, rw)
            self.assertEqual(len(self.ext.calls), 9)
            self.assertTrue(all(c["num_tokens"] == 1 for c in self.ext.calls))
            self.assertFalse(self.ext.poisoned)

        def test_status_snapshot(self):
            st = rp.moe_multi_token_status()
            self.assertTrue(st["on"])
            self.assertEqual(st["rows"], (2, 5))
            self.assertEqual(st["slots_max"], 128)
            self.assertFalse(st["require_cuda"])


    class TestRowLoopUnchanged(MTBase):

        def test_r1_declines_candidate(self):
            m, y, sel, rw, _t = self.make(1)
            ok, why = rp.moe_multi_token_supported(m, y, sel, rw)
            self.assertFalse(ok)
            self.assertIn("outside 2..5", why)

        def test_r1_loop_verbatim_shard_bounds(self):
            # the row loop keeps passing the SHARD bounds and raw global ids
            # with num_tokens == 1 -- verbatim v1.4.4 behaviour
            m, y, sel, rw, tables = self.make(1, mine=2, maxe=6, E=8,
                                              sel_rows=[[3, 5, 2, 4]])
            rw = torch.full((1, 4), 0.25).half().contiguous()
            rp.moe_mgemm_bszN(m, y, sel, rw)
            self.assertEqual([c["stage"] for c in self.ext.calls],
                             ["gate", "up", "down"])
            d = self.ext.calls[2]
            self.assertEqual(d["min_index"], 2)
            self.assertEqual(d["max_index"], 6)
            self.assertEqual(d["num_tokens"], 1)
            self.assertEqual(d["idx"], [3, 5, 2, 4])       # NOT pre-rebased
            self.assertFalse(self.ext.poisoned)
            exp = expected_rows(tables, y, sel, rw, 2, 6)
            got = m.experts_cfg.out_bszn[0].float()
            self.assertTrue(torch.allclose(got, exp[0], atol=5e-3),
                            (got - exp[0]).abs().max())

        def test_rows6_falls_back_to_loop(self):
            m, y, sel, rw, _t = self.make(6)
            self.assertFalse(rp.moe_multi_token_step(m, y, sel, rw))
            rp.moe_mgemm_bszN(m, y, sel, rw)
            self.assertEqual(len(self.ext.calls), 18)
            self.assertTrue(all(c["num_tokens"] == 1 for c in self.ext.calls))

        def test_gateless_uses_up_activation_down_candidate(self):
            m, y, sel, rw, _t = self.make(3, gated=False)
            ok, why = rp.moe_multi_token_supported(m, y, sel, rw)
            self.assertTrue(ok, why)
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            stages = [c["stage"] for c in self.ext.calls]
            self.assertNotIn("gate", stages)
            self.assertEqual(stages, ["up", "down"])


    class TestGateRejections(MTBase):

        def _reject(self, mod, y, sel, rw, want):
            ok, why = rp.moe_multi_token_supported(mod, y, sel, rw)
            self.assertFalse(ok, f"expected rejection containing {want!r}")
            self.assertIn(want, why)
            self.assertFalse(rp.moe_multi_token_step(mod, y, sel, rw))
            self.assertEqual(self.ext.calls, [])           # refusal runs nothing

        def test_rejections(self):
            m, y, sel, rw, _t = self.make(3)

            with self.subTest("rows 6"):
                y6 = (torch.randn(6, 128) * 0.25).half()
                sel6 = torch.randint(0, 8, (6, 4)).long()
                rw6 = torch.softmax(torch.randn(6, 4), dim=-1).half()
                self._reject(m, y6, sel6, rw6, "outside 2..5")

            with self.subTest("top_k cannot form a multi-token chunk"):
                y13 = (torch.randn(5, 128) * 0.25).half()
                sel13 = torch.randint(0, 8, (5, 65)).long()
                rw13 = torch.softmax(torch.randn(5, 65), dim=-1).half()
                self._reject(m, y13, sel13, rw13, "top_k")

            with self.subTest("routing weights shape mismatch"):
                self._reject(m, y, sel, rw[:, :3].contiguous(), "shape")

            with self.subTest("quant K 9"):
                m9 = _Mod(8)
                m9.multi_down.K = 9
                self.ext.register(m9)
                self._reject(m9, y, sel, rw, "outside 1..8")

            with self.subTest("out_bszn rows < r"):
                m_o = _Mod(8)
                m_o.experts_cfg.out_bszn = torch.zeros(1, 128)
                self.ext.register(m_o)
                self._reject(m_o, y, sel, rw, "rows < 3")

            with self.subTest("unsliced pointer table vs shard"):
                m_p = _Mod(3, mine=2, maxe=6)
                self.ext.register(m_p)
                ysh = (torch.randn(3, 128) * 0.25).half()
                selp = torch.tensor([[2, 3, 4], [5, 2, 3], [4, 5, 2]]).long()
                rwp = torch.full((3, 3), 1.0 / 3).half()
                self._reject(m_p, ysh, selp, rwp, "pointer table")

            with self.subTest("degenerate shard range"):
                m_d = _Mod(4, mine=4, maxe=2)
                self.ext.register(m_d)
                self._reject(m_d, ysh, selp, rwp, "degenerate")

            with self.subTest("malformed expert range sign"):
                m_m = _Mod(4, mine=-1, maxe=4)
                self.ext.register(m_m)
                self._reject(m_m, ysh, selp, rwp, "malformed")

            with self.subTest("half a range (None side)"):
                m_h = _Mod(8)
                m_h.experts_cfg.min_expert = None
                self.ext.register(m_h)
                self._reject(m_h, y, sel, rw, "both min_expert")

            with self.subTest("intermediate not %128"):
                m_i = _Mod(8, I=100)
                self.ext.register(m_i)
                self._reject(m_i, y, sel, rw, "128")

            with self.subTest("dtypes"):
                self._reject(m, y.float(), sel, rw, "dtypes")
                self._reject(m, y, sel.int(), rw, "dtypes")
                self._reject(m, y, sel, rw.float(), "dtypes")

            with self.subTest("input layouts"):
                self._reject(m, y, sel, rw.t().contiguous().t(), "non-contiguous")
                self._reject(m, y[:, :100].contiguous(), sel, rw, "hidden width")
                self._reject(m, y, sel.unsqueeze(0), rw, "2-D")

            with self.subTest("native scratch layout validity"):
                m_s = _Mod(8)
                m_s.experts_cfg.yh = torch.zeros(32, 2, 128, dtype=torch.half)
                self.ext.register(m_s)
                self._reject(m_s, y, sel, rw, "shape")
                m_nc = _Mod(8)
                # (32,1,128) with last-dim stride 32: genuinely non-contiguous
                m_nc.experts_cfg.yh = torch.zeros(128, 1, 32,
                                                  dtype=torch.half).permute(2, 1, 0)
                self.assertFalse(m_nc.experts_cfg.yh.is_contiguous())
                self.ext.register(m_nc)
                self._reject(m_nc, y, sel, rw, "contiguous")
                m_dt = _Mod(8)
                m_dt.experts_cfg.interm_u = torch.zeros(32, 1, 128, dtype=torch.float)
                self.ext.register(m_dt)
                self._reject(m_dt, y, sel, rw, "dtype")
                m_al = _Mod(8)
                m_al.experts_cfg.interm_a = m_al.experts_cfg.interm_g
                self.ext.register(m_al)
                self._reject(m_al, y, sel, rw, "aliases")
                m_ob = _Mod(8)
                m_ob.experts_cfg.out_bszn = torch.zeros(8, 1, 128)
                self.ext.register(m_ob)
                self._reject(m_ob, y, sel, rw, "[rows,width]")

            with self.subTest("require_cuda on CPU tensors"):
                rp._MOE_MT["require_cuda"] = True
                self._reject(m, y, sel, rw, "non-CUDA")

            with self.subTest("unloaded cfg (unload path)"):
                m_u = _Mod(8)
                m_u.experts_cfg = None
                self._reject(m_u, y, sel, rw, "experts_cfg unloaded")
                m_n = _Mod(8)
                m_n.multi_gate = None
                self.ext.register(m_n)
                self._reject(m_n, y, sel, rw, "MultiLinear tables absent")


    class TestCandidateUnsharded(MTBase):

        def test_slot_major_layout_and_numerics(self):
            m, y, sel, rw, tables = self.make(3, act_limit=7.0)
            r, k = sel.shape
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            self.assertEqual([c["stage"] for c in self.ext.calls],
                             ["gate", "up", "down"])
            for c in self.ext.calls:
                self.assertEqual(c["min_index"], -1)
                self.assertEqual(c["max_index"], -1)
                self.assertEqual(c["num_tokens"], r)
            g = self.ext.calls[0]
            self.assertEqual(tuple(g["A"].shape), (r * k, 1, y.shape[1]))
            for j in range(r * k):                  # each row repeated k times
                self.assertTrue(torch.equal(g["A"][j, 0], y[j // k]))
            d = self.ext.calls[2]
            self.assertEqual(d["idx"], sel.reshape(-1).tolist())   # no -1, order kept
            self.assertTrue(torch.equal(d["weights"], rw.reshape(-1)))
            self.assertIsNone(d["env_fuse"])        # unsharded: fused form kept
            self.assertFalse(self.ext.poisoned)
            self.assertEqual(self.ext.hazards, 0)
            self.assertEqual(m.act_calls, [7.0])    # act_limit passed through, once
            exp = expected_rows(tables, y, sel, rw, None, None)
            got = m.experts_cfg.out_bszn[:r].float()
            self.assertTrue(torch.allclose(got, exp, atol=5e-3),
                            (got - exp).abs().max())
            # one bulk copy: out_bszn[:r] == the reduced first r down rows
            native = getattr(m, "_rocm_moe_mt_scratch", {}).get("native")
            src = (native["out_d"] if native else m.experts_cfg.out_d)
            self.assertTrue(torch.equal(got, src[:r, 0, :].float()))

        def test_inputs_not_mutated(self):
            m, y, sel, rw, _t = self.make(4, mine=2, maxe=6, E=8,
                                          sel_rows=[[3, 3, 3, 3]] * 4)
            y0, sel0, rw0 = y.clone(), sel.clone(), rw.clone()
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            self.assertTrue(torch.equal(y, y0))
            self.assertTrue(torch.equal(sel, sel0))
            self.assertTrue(torch.equal(rw, rw0))


    class TestCandidateSharded(MTBase):

        def test_position_preserving_local_mask(self):
            m, y, sel, rw, tables = self.make(2, mine=2, maxe=6, E=8)
            # [3,1,5,2] -> [1,-1,3,0] ; [0,1,6,7] -> all outside (-1 x4)
            sel = torch.tensor([[3, 1, 5, 2], [0, 1, 6, 7]]).long()
            rw = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                               [0.4, 0.3, 0.2, 0.1]]).half().contiguous()
            m.experts_cfg.out_d.fill_(float("nan"))     # unsafe stale scratch
            m.experts_cfg.interm_g.fill_(float("nan"))
            m.experts_cfg.interm_u.fill_(float("nan"))
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            g = self.ext.calls[0]
            self.assertEqual(g["idx"], [1, -1, 3, 0, -1, -1, -1, -1])
            self.assertEqual(g["num_tokens"], 2)
            self.assertEqual(g["min_index"], -1)
            d = self.ext.calls[2]
            self.assertEqual(d["env_fuse"], "0")        # weighted call de-fused
            self.assertEqual(d["inactive_c_absmax_at_entry"], 0.0)
            self.assertFalse(self.ext.poisoned)
            self.assertEqual(self.ext.hazards, 0)
            exp = expected_rows(tables, y, sel, rw, 2, 6)
            got = m.experts_cfg.out_bszn[:2].float()
            self.assertTrue(torch.allclose(got[0], exp[0], atol=5e-3),
                            (got[0] - exp[0]).abs().max())
            # empty partial rank: exact zero row, and nothing NaN leaked
            self.assertTrue(torch.equal(got[1], torch.zeros(128)))
            self.assertTrue(torch.isfinite(got).all())

        def test_boundary_ids_resolve_locally(self):
            m, y, sel, rw, _t = self.make(2, mine=2, maxe=6, E=8,
                                          sel_rows=[[2, 1, 5, 6], [3, 4, 2, 5]])
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            # lo -> 0, lo-1 -> masked, hi-1 -> width-1, hi -> masked; second
            # row fully in range; all R*K slots and positions retained
            self.assertEqual(self.ext.calls[0]["idx"],
                             [0, -1, 3, -1, 1, 2, 0, 3])

        def test_env_toggle_absent_after_shard_run(self):
            m, y, sel, rw, _t = self.make(2, mine=2, maxe=6, E=8,
                                          sel_rows=[[7, 7, 7, 7], [7, 7, 7, 7]])
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            self.assertNotIn(FUSE_ENV, os.environ)      # restored to absent

        def test_env_restored_on_exception(self):
            for preset in (None, "1"):
                os.environ.pop(FUSE_ENV, None)
                if preset is not None:
                    os.environ[FUSE_ENV] = preset
                m, y, sel, rw, _t = self.make(2, mine=2, maxe=6, E=8)
                self.ext.raise_on_stage = "down"
                with self.assertRaises(RuntimeError):
                    rp.moe_multi_token_step(m, y, sel, rw)
                self.assertEqual(os.environ.get(FUSE_ENV), preset)
                self.ext.raise_on_stage = None
                self.ext.calls.clear()

        def test_stale_scratch_across_reused_runs(self):
            # A, B, A on one shard view: different mask positions must not
            # leak between runs (slot scratch is reused; zeroing is per call)
            m, y, sel, rw, tables = self.make(2, mine=2, maxe=6, E=8)
            selA = torch.tensor([[3, 1, 5, 2], [0, 2, 6, 7]]).long()
            rwA = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                                [0.4, 0.3, 0.2, 0.1]]).half().contiguous()
            self.assertTrue(rp.moe_multi_token_step(m, y, selA, rwA))
            outA = m.experts_cfg.out_bszn[:2].clone()
            selB = torch.tensor([[2, 2, 3, 3], [4, 4, 5, 5]]).long()
            rwB = torch.full((2, 4), 0.25).half()
            self.assertTrue(rp.moe_multi_token_step(m, y, selB, rwB))
            outB = m.experts_cfg.out_bszn[:2].clone()
            self.assertTrue(rp.moe_multi_token_step(m, y, selA, rwA))
            outA2 = m.experts_cfg.out_bszn[:2].clone()
            self.assertTrue(torch.equal(outA, outA2))                   # no stale leak
            self.assertFalse(torch.equal(outA, outB))                   # B really differed
            expA = expected_rows(tables, y, selA, rwA, 2, 6)
            self.assertTrue(torch.allclose(outA.float(), expA, atol=5e-3))
            expB = expected_rows(tables, y, selB, rwB, 2, 6)
            self.assertTrue(torch.allclose(outB.float(), expB, atol=5e-3))


    class TestFusedCounterHazardModel(MTBase):
        """The FakeExt hazard emulation itself, plus proof the candidate
        avoids it: negative controls so the sharded tests above cannot pass
        if the de-fuse workaround were removed from the route.

        Whether the de-fused masked sum is BIT-identical to the cooperative
        guarded sum on real hardware is NOT claimed here or anywhere -- the
        probe measures it; that is exactly why the route is default-OFF."""

        def setUp(self):
            super().setUp()
            self.mod, self.y, self.sel, self.rw, self.tables = \
                self.make(2, mine=2, maxe=6, E=8)
            self.cfg = self.mod.experts_cfg
            self.I = self.cfg.interm_a.shape[-1]

        def _masked_weighted_call(self, env=None,
                                  idx_vals=(0, -1, 3, 0, -1, -1, -1, -1)):
            # C is the module's own down scratch (stage resolves from the
            # multi_down pointer table, like the real kernel)
            C = self.cfg.out_d
            C.zero_()
            if env is None:
                os.environ.pop(FUSE_ENV, None)
            else:
                os.environ[FUSE_ENV] = env
            idx = torch.tensor([list(idx_vals)])
            w = torch.full((1, 8), 0.5).half()
            A = (torch.randn(8, 1, self.I) * 0.2).half()
            Ahad = torch.zeros(8, 1, self.I, dtype=torch.half)
            self.ext.exl3_mgemm(A, self.mod.multi_down.ptrs_trellis, C, None,
                                Ahad, None, idx, w, 4, -1, False, False,
                                -1, -1, 0, 2)
            return A

        def test_fused_masked_call_skips_reduce_and_poisons(self):
            A = self._masked_weighted_call(env=None)
            self.assertTrue(self.ext.poisoned)
            self.assertEqual(self.ext.hazards, 1)
            per_slot = A[0, 0].float() @ self.tables["down"][0] * 0.5
            self.assertTrue(torch.allclose(self.cfg.out_d[0, 0], per_slot, atol=1e-6))
            # a LATER, fully-valid weighted call must also skip the reduce:
            # the dirty arrival counters poison every subsequent call too
            A3 = self._masked_weighted_call(env="1",
                                            idx_vals=(0, 1, 2, 3, 0, 1, 2, 3))
            self.assertEqual(self.ext.hazards, 1)          # no new masked call
            per_slot3 = A3[0, 0].float() @ self.tables["down"][0] * 0.5
            self.assertTrue(torch.allclose(self.cfg.out_d[0, 0], per_slot3, atol=1e-6))

        def test_defused_masked_call_reduces_zeroed_inactive(self):
            A = self._masked_weighted_call(env="0")
            self.assertFalse(self.ext.poisoned)
            self.assertEqual(self.ext.hazards, 0)
            # token 0: slots 0,2,3 valid + slot 1 masked (zeroed row)
            want = torch.zeros(128)
            for slot, mat in ((0, 0), (2, 3), (3, 0)):
                want = want + A[slot, 0].float() @ self.tables["down"][mat] * 0.5
            self.assertTrue(torch.allclose(self.cfg.out_d[0, 0], want, atol=5e-4))
            # token 1: all four slots masked -> exact zero partial
            self.assertTrue(torch.equal(self.cfg.out_d[1, 0], torch.zeros(128)))

        def test_candidate_route_never_trips_the_hazard(self):
            m, y, sel, rw, _t = self.make(2, mine=2, maxe=6, E=8,
                                          sel_rows=[[3, 1, 5, 2], [0, 1, 6, 7]])
            os.environ[FUSE_ENV] = "1"          # hostile ambient setting
            try:
                self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            finally:
                if self.saved_env[0] is None:
                    os.environ.pop(FUSE_ENV, None)
                else:
                    os.environ[FUSE_ENV] = self.saved_env[0]
            self.assertFalse(self.ext.poisoned)  # the per-call de-fuse held
            self.assertEqual(self.ext.hazards, 0)
            got = m.experts_cfg.out_bszn[:2].float()
            self.assertTrue(torch.equal(got[1], torch.zeros(128)))


    class TestScratchOwnership(MTBase):

        def test_module_owned_bounded_scratch_reused(self):
            m, y, sel, rw, _t = self.make(3)
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            scratch = m._rocm_moe_mt_scratch
            self.assertNotIn("native", scratch)      # cfg rows 32 >= slots: cfg path
            self.assertEqual(tuple(scratch["hidden_slots"].shape), (128, 128))
            first = (id(scratch), scratch["hidden_slots"].data_ptr())
            other, y2, s2, w2, _t2 = self.make(4)    # a different module
            self.assertTrue(rp.moe_multi_token_step(other, y2, s2, w2))
            # fresh module -> its OWN dict (layers/target/draft cannot alias)
            self.assertNotEqual(id(other._rocm_moe_mt_scratch), first[0])
            y3 = (torch.randn(2, 128) * 0.25).half()
            sel3 = torch.randint(0, 8, (2, 4)).long()
            rw3 = torch.softmax(torch.randn(2, 4), dim=-1).half()
            self.assertTrue(rp.moe_multi_token_step(m, y3, sel3, rw3))
            again = m._rocm_moe_mt_scratch
            self.assertEqual((id(again), again["hidden_slots"].data_ptr()), first)
            # other modules never grew this module's dict either
            self.assertEqual(sorted(k for k in again if k != "native"),
                             ["cfg_sig", "device", "hidden", "hidden_slots",
                              "indices", "mask", "torch"])

        def test_native_bundle_for_short_cfg(self):
            # a loader whose experts_cfg rows are only top-k sized: the route
            # must allocate its own bounded 128-row native bundle, use it,
            # and stay numerically correct incl. shard masking
            m, y, sel, rw, tables = self.make(3, r_cap=2, mine=2, maxe=6, E=8)
            # 3 rows * 4 picks = 12 slots > r_cap*K = 8 cfg rows. out_bszn
            # stays cfg-owned (forward reads cfg.out_bszn[:bsz]): the
            # bundle replaces only the five slot-major native buffers.
            m.experts_cfg.out_bszn = torch.zeros(8, 128)
            self.ext.register(m)
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            scratch = m._rocm_moe_mt_scratch
            self.assertIn("native", scratch)
            for b in scratch["native"].values():
                self.assertEqual(b.shape[0], 128)   # bounded at the slot cap
            self.assertEqual([c["stage"] for c in self.ext.calls],
                             ["gate", "up", "down"])
            exp = expected_rows(tables, y, sel, rw, 2, 6)
            got = m.experts_cfg.out_bszn[:3].float()
            self.assertTrue(torch.allclose(got, exp[:3], atol=5e-3),
                            (got - exp[:3]).abs().max())
            self.assertFalse(self.ext.poisoned)

        def test_native_bundle_promotes_when_rows_grow(self):
            # The same module can first serve R=2 from its ordinary 8-slot
            # cfg, then receive R=5 (20 slots). The second call must promote
            # once instead of reusing an undersized ordinary view; later R=2
            # calls keep that promoted bundle.
            m, y2, sel2, rw2, _tables = self.make(2, r_cap=2)
            m.experts_cfg.out_bszn = torch.zeros(5, 128)
            self.ext.register(m)
            self.assertTrue(rp.moe_multi_token_step(m, y2, sel2, rw2))
            first = m._rocm_moe_mt_scratch
            self.assertNotIn("native", first)

            g = torch.Generator().manual_seed(904)
            y5 = (torch.randn(5, 128, generator=g) * 0.25).half()
            sel5 = torch.stack([torch.randperm(8, generator=g)[:4]
                                for _ in range(5)]).long()
            rw5 = torch.softmax(torch.randn(5, 4, generator=g), dim=-1).half()
            self.assertTrue(rp.moe_multi_token_step(m, y5, sel5, rw5))
            promoted = m._rocm_moe_mt_scratch
            self.assertIs(promoted, first)  # promotion mutates ownership once
            self.assertIn("native", promoted)
            native_ptrs = tuple(v.data_ptr() for v in promoted["native"].values())

            self.assertTrue(rp.moe_multi_token_step(m, y2, sel2, rw2))
            self.assertIs(m._rocm_moe_mt_scratch, promoted)
            self.assertEqual(tuple(v.data_ptr() for v in promoted["native"].values()),
                             native_ptrs)

        def test_row_aligned_chunk_boundary_for_large_decode_batch(self):
            # With top-k=10, the native 128-slot bound permits 12 rows per
            # launch. Twenty rows must become 12+8, retaining token order and
            # copying each chunk into its corresponding out_bszn rows.
            old_cap = rp._MOE_MT.get("max_rows", 5)
            rp._MOE_MT["max_rows"] = 24
            try:
                m, y, sel, rw, _tables = self.make(20, k=10, E=16, r_cap=24)
                ok, why = rp.moe_multi_token_supported(m, y, sel, rw)
                self.assertTrue(ok, why)
                self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
                self.assertEqual([c["stage"] for c in self.ext.calls],
                                 ["gate", "up", "down"] * 2)
                self.assertEqual([c["num_tokens"] for c in self.ext.calls],
                                 [12, 12, 12, 8, 8, 8])
                self.assertEqual([c["A"].shape[0] for c in self.ext.calls],
                                 [120, 120, 120, 80, 80, 80])
                self.assertTrue(torch.isfinite(m.experts_cfg.out_bszn[:20]).all())
            finally:
                rp._MOE_MT["max_rows"] = old_cap

        def test_scratch_clear_on_unload_path(self):
            m, y, sel, rw, _t = self.make(3)
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            old = m._rocm_moe_mt_scratch
            self.assertTrue(old)                    # populated
            rp.moe_multi_token_scratch_clear(m)     # what patched unload calls
            self.assertFalse(hasattr(m, "_rocm_moe_mt_scratch"))
            self.assertEqual(len(old), 0)           # contents released
            # next run allocates fresh bounded staging again
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            self.assertTrue(hasattr(m, "_rocm_moe_mt_scratch"))
            self.assertNotEqual(id(m._rocm_moe_mt_scratch), id(old))

        def test_scratch_invalidated_when_cfg_changes(self):
            m, y, sel, rw, _t = self.make(3)
            self.assertTrue(rp.moe_multi_token_step(m, y, sel, rw))
            first = m._rocm_moe_mt_scratch
            m.experts_cfg = _Cfg(2, 4, 128, 128, 128, -1, -1)   # now short
            m.experts_cfg.out_bszn = torch.zeros(8, 128)         # cfg-owned rows
            self.ext.register(m)
            m2y = (torch.randn(3, 128) * 0.25).half()
            self.assertTrue(rp.moe_multi_token_step(m, m2y, sel, rw))
            self.assertIsNot(m._rocm_moe_mt_scratch, first)
            self.assertIn("native", m._rocm_moe_mt_scratch)


if __name__ == "__main__":
    unittest.main()
