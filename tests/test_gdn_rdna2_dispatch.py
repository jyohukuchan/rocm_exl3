"""
CPU-only coverage for the RDNA2 prefill-backend dispatch in
exllamav3/modules/gated_delta_net_fn/gated_delta_rule.py.

On gfx103x (RDNA2) long GDN prefill defaults to the fused native recurrent kernel instead of
the vendored fla chunk kernels: the chunk path only compiles there after the FP32 BF16-dot
promotion, and at the real Qwen shapes it is measurably slower than native at every size
(runs/qwen38/gdn-backend-compare-peak.json). EXL3_RDNA2_GDN_CHUNK opts RDNA2 back into the
chunk path. These tests pin the dispatch: the default per hardware, the override semantics
(documented as ``0`` off / anything else on), that the probe sees the tensors' own device,
that decode / short-prefill / history never evaluate the probe at all (hot-path guard), and
that the KDA branch is not gated. No kernel or CUDA device is touched: the vendor chunk
entry, the ext recurrent kernel and the hardware probe are all recording mocks, and the
native-vs-chunk output dtype (bf16) semantics are asserted through the mocks' return path.

Module loading mirrors tests/test_fla_rdna2_dispatch.py: in a full environment (built ext,
triton) the real module is imported; otherwise gated_delta_rule.py is executed standalone
against a stub parent tree (ext / util.tensor) that is removed from sys.modules again, so no
stub leaks into other test files. The per-test chunk recorder is likewise installed under
"exllamav3.vendor.fla" only for the duration of the test and restored after.
"""
import importlib.util
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODNAME = "exllamav3.modules.gated_delta_net_fn.gated_delta_rule"
VENDNAME = "exllamav3.vendor.fla"


def _load_gdn_module():
    try:
        from exllamav3.modules.gated_delta_net_fn import gated_delta_rule as gdn
        return gdn
    except Exception:
        pass  # no built ext / no triton on this machine: execute the real file against stubs
    made = {}

    def stub(pkg_name, attrs = None):
        mod = types.ModuleType(pkg_name)
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        made[pkg_name] = sys.modules.get(pkg_name, ...)   # keep any partial real import intact
        sys.modules[pkg_name] = mod
        return mod

    try:
        stub("exllamav3"); stub("exllamav3.modules"); stub("exllamav3.modules.gated_delta_net_fn")
        stub("exllamav3.util")
        stub("exllamav3.util.tensor", {
            "get_for_device": lambda input_dict, key, device, default = None: input_dict.get(key, default),
            "buffered_arange": lambda n, device = None: torch.arange(n, dtype = torch.int64, device = device),
        })
        stub("exllamav3.ext", {"exllamav3_ext": types.SimpleNamespace(
            cuda_recurrent_gated_delta_rule = None)})   # monkeypatched by the native fixture
        spec = importlib.util.spec_from_file_location(
            MODNAME, os.path.join(ROOT, "exllamav3", "modules", "gated_delta_net_fn", "gated_delta_rule.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # relative imports hit the stub sys.modules entries
    finally:
        for pkg_name, saved in made.items():
            if saved is ...:
                sys.modules.pop(pkg_name, None)
            else:
                sys.modules[pkg_name] = saved
        sys.modules.pop(MODNAME, None)  # keep the whole stub tree out of later test files
    return mod


gdn = _load_gdn_module()


# --------------------------------------------------------------------------- recording mocks

@pytest.fixture
def probe(monkeypatch):
    """Replace the hardware-probe seam of the module; returns a setter giving (answer, calls)"""
    def apply(rdna2: bool):
        calls = []
        def fake_device_lacks(device):
            calls.append(device)
            return rdna2
        monkeypatch.setattr(gdn, "_device_lacks_bf16_dot", fake_device_lacks)
        return calls
    return apply


@pytest.fixture
def native(monkeypatch):
    """Record every fused native recurrent kernel call and write a sentinel through the
    output-pointer argument, standing in for the ext kernel's in-place update"""
    calls = []

    def fake_recurrent(mixed_qkv, g, beta, state, out, num_k_heads, num_v_heads,
                       k_head_dim, v_head_dim, slots, history):
        calls.append(dict(mixed_qkv = mixed_qkv, g = g, beta = beta, state = state, out = out,
                          num_k_heads = num_k_heads, num_v_heads = num_v_heads,
                          k_head_dim = k_head_dim, v_head_dim = v_head_dim,
                          slots = slots, history = history))
        out.fill_(0.5)
        return None

    monkeypatch.setattr(gdn.ext, "cuda_recurrent_gated_delta_rule", fake_recurrent)
    return calls


@pytest.fixture
def chunk(monkeypatch):
    """Install the vendor package entry points as recorders for the duration of the test only
    (and restore any previously imported real package afterwards), so neither world leaks"""
    calls = {"gdn": [], "kda": []}

    def make(bucket):
        def fake(q, k, v, g = None, beta = None, initial_state = None,
                 output_final_state = False, use_qk_l2norm_in_kernel = False, **kwargs):
            bucket.append(dict(q = q, k = k, v = v, g = g, beta = beta, initial_state = initial_state,
                               output_final_state = output_final_state,
                               use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel))
            o = torch.zeros(q.shape[0], q.shape[1], v.shape[2], v.shape[3],
                            dtype = q.dtype, device = q.device)   # chunk returns q's original dtype
            s = torch.zeros(v.shape[2], v.shape[3], dtype = torch.float32, device = q.device) \
                if output_final_state else None
            return o, s
        return fake

    vend = types.ModuleType(VENDNAME)
    vend.chunk_gated_delta_rule = make(calls["gdn"])
    vend.chunk_kda = make(calls["kda"])
    saved = sys.modules.get(VENDNAME, ...)
    sys.modules[VENDNAME] = vend
    try:
        yield calls
    finally:
        if saved is ...:
            sys.modules.pop(VENDNAME, None)
        else:
            sys.modules[VENDNAME] = saved


# --------------------------------------------------------------------------- call harness

# tiny GDN geometry; num_v_heads 8 keeps the seqlen >= num_v_heads chunk-vs-native threshold
# testable in both directions (long prefill T = 33, short prefill T = 4, decode T = 1)
H, HV, DK, DV = 2, 8, 64, 32
KDIM, VDIM = H * DK, HV * DV


def _gdn_inputs(bsz, T, dtype = torch.bfloat16):
    torch.manual_seed(T * bsz + 1)
    mixed_qkv = torch.randn(bsz, T, 2 * KDIM + VDIM, dtype = dtype) * 0.25
    g = torch.randn(bsz, T, HV) * 0.5 - 1
    beta = torch.sigmoid(torch.randn(bsz, T, HV, dtype = dtype))
    slots = torch.arange(bsz, dtype = torch.int64)
    return dict(mixed_qkv = mixed_qkv, beta = beta, g = g, recurrent_state = None,
                recurrent_slots = slots, history = False, save_state = False,
                num_k_heads = H, num_v_heads = HV, k_dim = KDIM, v_dim = VDIM,
                k_head_dim = DK, v_head_dim = DV)


def _kda_inputs(bsz, T):
    kw = _gdn_inputs(bsz, T)
    kw["g"] = torch.randn(bsz, T, HV, DK)          # per-k-channel decay
    kw["channelwise_g"] = True
    return kw


# --------------------------------------------------------------------------- dispatch (GDN)

def test_long_prefill_goes_to_chunk_when_probe_says_not_rdna2(probe, chunk):
    probed = probe(False)
    kw = _gdn_inputs(2, 33)
    out = gdn.gated_delta_rule_fn(**kw)
    assert probed == [kw["mixed_qkv"].device], "the tensors' own device is asked, once per prefill"
    assert len(chunk["gdn"]) == 2, "one chunk call per recurrent slot"
    for call in chunk["gdn"]:
        assert call["q"].dtype == torch.bfloat16 and call["beta"].dtype == torch.bfloat16
        assert call["use_qk_l2norm_in_kernel"] is True
    assert out.dtype == torch.bfloat16 and out.shape == (2, 33, HV, DV)


def test_long_prefill_defaults_to_native_on_rdna2(probe, native):
    probed = probe(True)
    kw = _gdn_inputs(2, 33)
    out = gdn.gated_delta_rule_fn(**kw)   # no chunk fixture: the chunk branch must not be entered
    assert probed == [kw["mixed_qkv"].device]
    assert len(native) == 1, "the fused recurrent kernel serves the whole batch in one call"
    call = native[0]
    assert call["mixed_qkv"] is kw["mixed_qkv"] and call["g"] is kw["g"] and call["beta"] is kw["beta"]
    assert call["slots"] is kw["recurrent_slots"] and call["history"] is False
    assert call["out"].dtype == torch.bfloat16 and call["out"].shape == (2, 33, HV, DV)
    assert call["state"].dtype == torch.float32 and call["state"].shape == (2, 1, HV, DK, DV)
    assert torch.equal(out, torch.full_like(out, 0.5)), "the kernel's output buffer is what is returned"
    assert out.dtype == torch.bfloat16, "bf16 output semantics identical to the short-prefill native path"


def test_read_only_initial_state_is_not_sent_to_mutating_native_kernel(probe, chunk, native):
    probe(True)
    kw = _gdn_inputs(1, 33)
    initial = torch.randn(1, 1, HV, DK, DV)
    before = initial.clone()
    kw.update(recurrent_state=initial, save_state=False)
    gdn.gated_delta_rule_fn(**kw)
    assert not native and len(chunk["gdn"]) == 1
    assert torch.equal(initial, before)


def test_rdna2_env_override_forces_chunk_path(probe, chunk, monkeypatch):
    monkeypatch.setenv("EXL3_RDNA2_GDN_CHUNK", "1")
    probed = probe(True)
    kw = _gdn_inputs(2, 33)
    out = gdn.gated_delta_rule_fn(**kw)
    assert len(chunk["gdn"]) == 2, "EXL3_RDNA2_GDN_CHUNK=1 opts RDNA2 back into the FP32 chunk path"
    assert probed == [], "the override short-circuits before the probe: no device query at all"
    assert out.dtype == torch.bfloat16


@pytest.mark.parametrize("value", ["1", "true", "on"])
def test_override_semantics_zero_off_anything_else_on(probe, chunk, monkeypatch, value):
    # env_vars.md convention: 0 is off, any other value is on
    monkeypatch.setenv("EXL3_RDNA2_GDN_CHUNK", value)
    probe(True)
    gdn.gated_delta_rule_fn(**_gdn_inputs(1, 33))
    assert len(chunk["gdn"]) == 1, f"EXL3_RDNA2_GDN_CHUNK={value} must force the chunk path on RDNA2"


@pytest.mark.parametrize("value", [None, "0"])
def test_rdna2_default_and_explicit_zero_use_native(probe, native, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("EXL3_RDNA2_GDN_CHUNK", raising = False)
    else:
        monkeypatch.setenv("EXL3_RDNA2_GDN_CHUNK", value)
    probe(True)
    gdn.gated_delta_rule_fn(**_gdn_inputs(1, 33))
    assert len(native) == 1


def test_override_cannot_send_other_hardware_to_native(probe, chunk, native):
    # the override only ever opts INTO chunk on RDNA2; it must never flip non-RDNA2 to native
    probe(False)
    gdn.gated_delta_rule_fn(**_gdn_inputs(1, 33))
    assert len(chunk["gdn"]) == 1 and native == []


# --------------------------------------------------------------------------- hot-path guard

def test_decode_and_short_prefill_never_evaluate_the_probe(probe, native):
    probed = probe(True)      # would pick native everywhere if consulted: assert it never is
    kw = _gdn_inputs(1, 1)    # decode-length
    kw["history"] = True
    gdn.gated_delta_rule_fn(**kw)
    gdn.gated_delta_rule_fn(**_gdn_inputs(1, HV - 1))   # short prefill, no history
    assert probed == [], "the chunk-vs-native predicate must not touch the vendor import or the " \
                         "device probe on the ordinary short-token decode path"
    assert len(native) == 2


def test_history_long_prefill_stays_native_without_probe_on_rdna2(probe, native):
    probed = probe(True)
    kw = _gdn_inputs(1, 33)
    kw["history"] = True
    gdn.gated_delta_rule_fn(**kw)
    assert probed == [] and len(native) == 1, "history semantics untouched: chunk was never eligible"


# --------------------------------------------------------------------------- KDA isolation

def test_kda_long_prefill_is_not_gated_on_rdna2(probe, chunk):
    probed = probe(True)
    out = gdn.gated_delta_rule_fn(**_kda_inputs(1, 33))
    assert probed == [], "the RDNA2 dispatch is scoped to the non-channelwise GDN branch"
    assert len(chunk["kda"]) == 1 and chunk["gdn"] == []
    assert out.dtype == torch.bfloat16
