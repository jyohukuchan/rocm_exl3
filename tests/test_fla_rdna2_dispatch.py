"""
CPU-only coverage for the gfx103x (RDNA2) BF16 compatibility path of the vendored fla package
(exllamav3/vendor/fla). RDNA2 has no BF16 dot instruction, so a Triton tl.dot on BF16 operands
fails LLVM selection (%llvm.amdgcn.fdot2.bf16.bf16) and aborts the process while compiling;
chunk_gated_delta_rule now runs BF16 operands through the chunk path in FP32 on those parts and
casts the output back to the original dtype. These tests pin that dispatch: which hardware and
which dtypes trigger the promotion, that gates and the recurrent state are left alone, that the
KDA / simple-GLA entries and non-RDNA2 calls are untouched, and that the output keeps the
caller's dtype and device.

Nothing launches a kernel: the arch-string test is pure, the device probe is exercised through a
spied torch.cuda.get_device_properties, and the entry-point test replaces every kernel function
chunk_gated_delta_rule calls with a dtype-recording mock. If triton is not installed, the
vendored package is loaded standalone (under a private name, so the heavy exllamav3 package
__init__ is not required) with a minimal import-time triton stub; if it is installed and the
real package imports, the same tests run against it unmodified.
"""
import importlib.util
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _install_triton_stub():
    """
    Stand-in for the parts of triton's surface the vendored modules touch at import time
    (decorators, Config, the driver target, language attributes). Kernel bodies are never
    executed: every entry point that would launch one is mocked in the tests below.
    """
    triton = types.ModuleType("triton")
    triton.__version__ = "3.4.0"
    triton.Config = lambda *args, **kwargs: None

    def jit(fn = None, **kwargs):
        return fn if fn is not None else lambda f: f

    def autotune(*args, **kwargs):
        assert not args, "stub triton.autotune only models the decorator-with-arguments form"
        return lambda fn: fn

    triton.jit = jit
    triton.autotune = autotune
    triton.heuristics = lambda values: (lambda fn: fn)
    triton.cdiv = lambda a, b: -(-a // b)
    triton.next_power_of_2 = lambda n: 1 << (max(int(n), 1) - 1).bit_length()
    triton.set_allocator = lambda fn: None

    class _AnyAttr(types.ModuleType):
        """triton.language: any attribute access yields a callable placeholder (constexpr tags
        are constructed, and tl.dot/tl.load... would be called, at module or kernel scope)"""
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return lambda *args, **kwargs: None

    tl = _AnyAttr("triton.language")
    extra = _AnyAttr("triton.language.extra")
    libdevice = _AnyAttr("triton.language.extra.libdevice")
    extra.libdevice = libdevice
    tl.extra = extra
    triton.language = tl
    triton.runtime = types.SimpleNamespace(driver = types.SimpleNamespace(
        active = types.SimpleNamespace(
            get_current_target = lambda: types.SimpleNamespace(backend = "hip"),
            utils = types.SimpleNamespace(
                get_device_properties = lambda i: {"multiprocessor_count": 1, "max_shared_mem": 101376}),
        )))
    for name, mod in (("triton", triton), ("triton.language", tl),
                      ("triton.language.extra", extra), ("triton.language.extra.libdevice", libdevice)):
        sys.modules[name] = mod


def _load_vendored_fla():
    try:
        import exllamav3.vendor.fla as vend
        return vend, sys.modules[vend.__name__ + ".utils"]
    except Exception:
        pass  # no built ext and/or no triton on this machine: load the package standalone
    stubbed = importlib.util.find_spec("triton") is None
    if stubbed:
        _install_triton_stub()
    pkg_dir = os.path.join(ROOT, "exllamav3", "vendor", "fla")
    name = "vendored_fla_under_test"
    spec = importlib.util.spec_from_file_location(name, os.path.join(pkg_dir, "__init__.py"),
                                                  submodule_search_locations = [pkg_dir])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        # Keep the private package's references, but don't expose our import-only
        # Triton stand-ins to unrelated tests collected later in this process.
        if stubbed:
            for module_name in list(sys.modules):
                if module_name == "triton" or module_name.startswith("triton."):
                    del sys.modules[module_name]
    return mod, sys.modules[name + ".utils"]


vend, vend_utils = _load_vendored_fla()


# --------------------------------------------------------------------------- arch string (pure)

@pytest.mark.parametrize("name", [
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036", "gfx1037",
    "GFX1030",                                    # case of the vendor spelling is not contractual
    "gfx1030:sramecc+:xnack-",                    # feature suffixes some ROCm stacks append
    "gfx1030 (RADV NAVI21)",                      # the RADV suffix Mesa-based builds report
])
def test_is_amd_rdna2_arch_matches_gfx103x(name):
    assert vend_utils.is_amd_rdna2_arch(name)


@pytest.mark.parametrize("name", [
    "", "gfx103", "gfx1010", "gfx1012",            # incomplete, and RDNA1
    "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151",   # RDNA3 / 3.5: unaffected
    "gfx1200", "gfx1201",                          # RDNA4: unaffected
    "gfx900", "gfx90a", "gfx942", "gfx950",        # CDNA: unaffected
    "sm_89", "Ada Lovelace",                       # NVIDIA strings: unaffected
])
def test_is_amd_rdna2_arch_rejects_everything_else(name):
    assert not vend_utils.is_amd_rdna2_arch(name)


# --------------------------------------------------------------------------- device probe (host)

@pytest.fixture
def probe(monkeypatch):
    """Pretend the driver reports per-index gcnArchNames, and count how often (and for which
    index) the probe actually reads device properties"""
    vend_utils._arch_lacks_bf16_dot.cache_clear()
    queries = []
    arch_names = {}

    class _Props:
        def __init__(self, name):
            self.gcnArchName = name

    def fake_properties(index):
        queries.append(index)
        return _Props(arch_names.get(index, "gfx9999"))

    monkeypatch.setattr(torch.cuda, "get_device_properties", fake_properties)
    monkeypatch.setattr(vend_utils, "IS_AMD", True)
    yield types.SimpleNamespace(queries = queries, arch_names = arch_names)
    vend_utils._arch_lacks_bf16_dot.cache_clear()


def test_probe_queries_the_requested_device_index(probe):
    probe.arch_names[3] = "gfx1030 (RADV NAVI21)"
    probe.arch_names[1] = "gfx1151"
    assert vend_utils.device_lacks_bf16_dot(torch.device("cuda:3")) is True
    assert vend_utils.device_lacks_bf16_dot(torch.device("cuda:1")) is False
    assert probe.queries == [3, 1], "must read the tensor's own device, not device 0 / the current one"


def test_probe_caches_per_device_index(probe):
    probe.arch_names[2] = "gfx1032"
    for _ in range(4):
        assert vend_utils.device_lacks_bf16_dot(torch.device("cuda:2")) is True
        assert vend_utils.device_lacks_bf16_dot(torch.device("cuda:0")) is False
    assert probe.queries == [2, 0], "one properties read per index for the process lifetime, so the steady state per forward is a dict lookup (no launch, no sync)"


def test_probe_resolves_indexless_cuda_device(probe, monkeypatch):
    probe.arch_names[2] = "gfx1034"
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    assert vend_utils.device_lacks_bf16_dot(torch.device("cuda")) is True
    assert probe.queries == [2]


def test_probe_never_touches_the_driver_for_non_cuda_devices(probe):
    assert vend_utils.device_lacks_bf16_dot(torch.device("cpu")) is False
    assert vend_utils.device_lacks_bf16_dot(torch.device("meta")) is False
    assert probe.queries == []


def test_probe_is_false_off_amd_platforms(probe, monkeypatch):
    probe.arch_names[0] = "gfx1030"
    monkeypatch.setattr(vend_utils, "IS_AMD", False)   # NVIDIA / anything non-HIP
    assert vend_utils.device_lacks_bf16_dot(torch.device("cuda:0")) is False
    assert probe.queries == [], "NVIDIA hosts must not even read device properties"


def test_probe_is_false_when_the_property_read_fails(probe, monkeypatch):
    def boom(index):
        raise RuntimeError("no driver")
    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert vend_utils.device_lacks_bf16_dot(torch.device("cuda:0")) is False


# --------------------------------------------------------------------------- promotion helper

def _quad(dtype, beta_dtype = None):
    B, T, H, HV, K, V = 1, 8, 2, 4, 8, 8
    q = torch.zeros(B, T, H, K, dtype = dtype)
    k = torch.zeros(B, T, H, K, dtype = dtype)
    v = torch.zeros(B, T, HV, V, dtype = dtype)
    beta = torch.zeros(B, T, HV, dtype = beta_dtype or dtype)
    return q, k, v, beta


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("promote", [False, True])
def test_promote_helper_casts_only_bf16_and_keeps_out_dtype(dtype, promote):
    q, k, v, beta = _quad(dtype)
    qs, ks, vs, betas, out_dtype = vend._promote_bf16_operands(q, k, v, beta, promote)
    expected = torch.float32 if (promote and dtype == torch.bfloat16) else dtype
    assert (qs.dtype, ks.dtype, vs.dtype, betas.dtype) == (expected,) * 4
    assert out_dtype == dtype, "the caller's original dtype is what the output must come back as"
    if expected == dtype:
        assert all(a is b for a, b in zip((qs, ks, vs, betas), (q, k, v, beta))), "a no-op must not copy"


def test_promote_helper_leaves_non_bf16_operands_alone_under_promotion():
    q, k, v, _ = _quad(torch.bfloat16)
    beta = torch.zeros(1, 8, 4, dtype = torch.float32)   # the FP32-beta form of the probe script
    qs, ks, vs, betas, out_dtype = vend._promote_bf16_operands(q, k, v, beta, promote = True)
    assert betas is beta, "already FP32: passed through"
    assert (qs.dtype, ks.dtype, vs.dtype) == (torch.float32,) * 3
    assert out_dtype == torch.bfloat16


# --------------------------------------------------------------------------- entry-point dispatch

def _gdn_inputs(dtype = torch.bfloat16, beta_dtype = None):
    # B1 T65 mirrors the failing shape class (a partial final chunk); heads/dims shrunk for speed
    B, T, H, HV, K, V = 1, 65, 2, 4, 8, 8
    q = torch.randn(B, T, H, K, dtype = dtype) * 0.25
    k = torch.randn(B, T, H, K, dtype = dtype) * 0.25
    v = torch.randn(B, T, HV, V, dtype = dtype) * 0.25
    g = torch.randn(B, T, HV) * 0.5 - 1          # fp32 log-space decay, per the contract
    beta = torch.sigmoid(torch.randn(B, T, HV, dtype = beta_dtype or dtype))
    state = torch.randn(B, HV, K, V) * 0.05      # fp32 recurrent state, per the contract
    return q, k, v, g, beta, state


@pytest.fixture
def kernels(monkeypatch):
    """Replace every kernel entry point the three prefill wrappers launch with a mock that
    records the dtypes it received and returns dtype-faithful stand-ins (the real code derives
    its buffers from the operands' dtypes the same way)"""
    seen = {}

    def record(stage, **tensors):
        seen[stage] = {name: (t.dtype if t is not None else None) for name, t in tensors.items()}

    def l2norm_fwd(x, *args, **kwargs):
        record("l2norm", x = x)
        return x, torch.zeros(x.shape[:-1], dtype = torch.float32, device = x.device)

    def chunk_local_cumsum(g, **kwargs):
        record("cumsum", g = g)
        return torch.zeros_like(g, dtype = torch.float32)   # the real one emits fp32 by contract

    def chunk_gated_delta_rule_fwd_intra(k = None, v = None, g = None, beta = None, **kwargs):
        record("intra", k = k, v = v, g = g, beta = beta)
        return k.new_zeros(1), v.new_zeros(v.shape), k.new_zeros(1)

    def chunk_gated_delta_rule_fwd_h(k = None, w = None, u = None, initial_state = None,
                                     output_final_state = False, **kwargs):
        record("delta_h", k = k, u = u, initial_state = initial_state)
        final = (torch.zeros(1, dtype = torch.float32, device = k.device)
                 if output_final_state else None)
        return k.new_zeros(1), u.new_zeros(u.shape), final

    def chunk_fwd_o(q = None, k = None, v = None, **kwargs):
        record("o", q = q, v = v)
        return v.new_zeros(v.shape)

    def chunk_kda_fwd_intra(q = None, k = None, v = None, gk = None, beta = None, **kwargs):
        record("kda_intra", q = q, k = k, v = v, gk = gk, beta = beta)
        w = k.new_zeros(1); u = v.new_zeros(v.shape); kg = k.new_zeros(1)
        return w, u, k.new_zeros(1), kg, k.new_zeros(1), k.new_zeros(1)

    def chunk_gla_fwd_o_gk(q = None, v = None, g = None, A = None, h = None, **kwargs):
        record("gla_o", q = q, v = v)
        return v.new_zeros(v.shape)

    def chunk_fwd_h(k = None, v = None, g = None, h0 = None, output_final_state = False, **kwargs):
        record("chunk_h", k = k, v = v, h0 = h0)
        h = k.new_zeros(1)
        ht = v.new_zeros(v.shape) if output_final_state else None   # states_in_fp32=False in the caller
        return h, ht

    for fname, fn in dict(
        l2norm_fwd = l2norm_fwd,
        chunk_local_cumsum = chunk_local_cumsum,
        chunk_gated_delta_rule_fwd_intra = chunk_gated_delta_rule_fwd_intra,
        chunk_gated_delta_rule_fwd_h = chunk_gated_delta_rule_fwd_h,
        chunk_fwd_o = chunk_fwd_o,
        chunk_kda_fwd_intra = chunk_kda_fwd_intra,
        chunk_gla_fwd_o_gk = chunk_gla_fwd_o_gk,
        chunk_fwd_h = chunk_fwd_h,
    ).items():
        monkeypatch.setattr(vend, fname, fn)
    return types.SimpleNamespace(seen = seen)


@pytest.fixture
def patch_probe(monkeypatch):
    """Answer the per-device probe with a fixed value and record which device was asked"""
    def apply(rdna2: bool):
        probed = []
        def fake(device):
            probed.append(device)
            return rdna2
        monkeypatch.setattr(vend, "device_lacks_bf16_dot", fake)
        return probed
    return apply


def _run_gdn(q, k, v, g, beta, state, use_l2norm = True):
    return vend.chunk_gated_delta_rule(
        q, k, v, g = g, beta = beta, initial_state = state,
        output_final_state = True, use_qk_l2norm_in_kernel = use_l2norm)


def test_rdna2_bf16_promotes_operands_before_l2norm_and_keeps_output_dtype(patch_probe, kernels):
    probed = patch_probe(True)
    q, k, v, g, beta, state = _gdn_inputs(torch.bfloat16)
    o, final = _run_gdn(q, k, v, g, beta, state)

    assert probed == [q.device], "the probe is asked about the tensors' own device, once per call"
    assert kernels.seen["l2norm"]["x"] == torch.float32
    assert kernels.seen["intra"] == {"k": torch.float32, "v": torch.float32,
                                     "g": torch.float32, "beta": torch.float32}
    assert kernels.seen["delta_h"]["k"] == torch.float32
    assert kernels.seen["o"]["q"] == torch.float32
    assert o.dtype == torch.bfloat16 and o.shape == v.shape and o.device == q.device
    assert final.dtype == torch.float32


def test_rdna2_bf16_promotes_without_in_kernel_l2norm(patch_probe, kernels):
    patch_probe(True)
    q, k, v, g, beta, state = _gdn_inputs(torch.bfloat16)
    o, _ = _run_gdn(q, k, v, g, beta, state, use_l2norm = False)
    assert "l2norm" not in kernels.seen
    assert kernels.seen["intra"]["k"] == torch.float32
    assert o.dtype == torch.bfloat16


def test_rdna2_leaves_fp16_and_fp32_inputs_untouched_and_unprobed(patch_probe, kernels):
    probed = patch_probe(True)
    for dtype in (torch.float16, torch.float32):
        q, k, v, g, beta, state = _gdn_inputs(dtype)
        o, _ = _run_gdn(q, k, v, g, beta, state)
        assert kernels.seen["intra"] == {"k": dtype, "v": dtype, "g": torch.float32, "beta": dtype}
        assert o.dtype == dtype, "no FP16 downcast, no needless FP32 copies"
    assert probed == [], "non-BF16 calls never even consult the hardware probe"


def test_rdna2_with_fp32_beta_promotes_only_qkv(patch_probe, kernels):
    patch_probe(True)
    q, k, v, g, beta, state = _gdn_inputs(torch.bfloat16, beta_dtype = torch.float32)
    o, _ = _run_gdn(q, k, v, g, beta, state)
    assert kernels.seen["intra"] == {"k": torch.float32, "v": torch.float32,
                                     "g": torch.float32, "beta": torch.float32}
    assert o.dtype == torch.bfloat16


def test_gates_and_recurrent_state_keep_fp32_semantics(patch_probe, kernels):
    patch_probe(True)
    q, k, v, g, beta, state = _gdn_inputs(torch.bfloat16)
    _run_gdn(q, k, v, g, beta, state)
    assert kernels.seen["cumsum"]["g"] == torch.float32, "gate dtype is passed through unchanged"
    assert kernels.seen["delta_h"]["initial_state"] == torch.float32


def test_non_rdna2_bf16_path_is_unchanged(patch_probe, kernels):
    probed = patch_probe(False)
    q, k, v, g, beta, state = _gdn_inputs(torch.bfloat16)
    o, _ = _run_gdn(q, k, v, g, beta, state)
    assert probed == [q.device]
    assert kernels.seen["l2norm"]["x"] == torch.bfloat16
    assert kernels.seen["intra"] == {"k": torch.bfloat16, "v": torch.bfloat16,
                                     "g": torch.float32, "beta": torch.bfloat16}
    assert kernels.seen["o"]["v"] == torch.bfloat16
    assert o.dtype == torch.bfloat16, "identical to the pre-fix behavior on NVIDIA / RDNA3+ / RDNA4"


def test_kda_entry_is_not_promoted_even_on_rdna2(patch_probe, kernels):
    probed = patch_probe(True)
    q, k, v, g, beta, state = _gdn_inputs(torch.bfloat16)
    gk = torch.randn(1, 65, 4, 8)                     # KDA wants [B, T, HV, K] fp32 gates
    o, _ = vend.chunk_kda(q, k, v, g = gk, beta = beta, initial_state = state,
                          output_final_state = True, use_qk_l2norm_in_kernel = True)
    assert probed == [], "the fix is scoped to chunk_gated_delta_rule"
    assert kernels.seen["kda_intra"]["q"] == torch.bfloat16
    assert kernels.seen["delta_h"]["k"] == torch.bfloat16
    assert o.dtype == torch.bfloat16


def test_simple_gla_entry_is_not_promoted_even_on_rdna2(patch_probe, kernels):
    probed = patch_probe(True)
    B, T, H, K, V = 1, 65, 4, 8, 8
    q = torch.randn(B, T, H, K, dtype = torch.bfloat16)
    k = torch.randn(B, T, H, K, dtype = torch.bfloat16)
    v = torch.randn(B, T, H, V, dtype = torch.bfloat16)
    g = torch.randn(B, T, H)
    o, _ = vend.chunk_simple_gla(q, k, v, g = g, scale = 1.0)
    assert probed == []
    assert kernels.seen["chunk_h"]["k"] == torch.bfloat16
    assert o.dtype == torch.bfloat16
