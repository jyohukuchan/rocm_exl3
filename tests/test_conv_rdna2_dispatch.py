"""
CPU-only coverage for the RDNA2 length rule in
exllamav3/modules/gated_delta_net_fn/conv1d.py.

On gfx103x (RDNA2) the Triton conv kernels cannot compile at prefill-sized blocks: the bf16
elementwise accumulate lowers to llvm.amdgcn.fdot2.bf16.bf16, whose LLVM selection aborts the
process (root-verified on the V620 at T=33, runs/qwen38/conv33-repro.log). The native
conv1d_update_kernel loops over ANY seqlen -- C++ TORCH_CHECKs only K <= 16, dtypes, shapes and
contiguity -- and is pre-compiled (no Triton JIT), so eligible bf16 geometry takes the native
kernel at every length on that hardware ONLY. These tests pin the dispatch:

  * T <= MAX_CUDA_SEQLEN (32) goes native on every device exactly as before and NEVER evaluates
    the hardware probe (hot-path guard: no import, no device query at decode),
  * long prefill on gfx103x goes native with unchanged slot/history/output semantics,
  * long prefill on anything the probe does not flag (RDNA3/4, CDNA, NVIDIA) keeps the 32
    threshold and stays on the Triton path,
  * non-eligible dtype/shape (fp16/fp32 tensors, K > 16, CPU inputs) falls back to Triton even
    on RDNA2 -- and is decided before any probe is evaluated,
  * the probe is asked for the tensor's OWN device (cuda:1 inputs query cuda:1, never the
    current device or device 0) and answers are per-device, not a process-wide latch.

No kernel and no CUDA device is touched: the hardware probe, the ext native entry point and the
module's Triton path are all recording mocks, and output allocations run against a host-side
torch proxy. Native-kernel numerical correctness at 33/64/257/2048 is the root's GPU job; these
tests prove only WHICH path is chosen. No speedup is claimed or implied.

Module loading mirrors tests/test_gdn_rdna2_dispatch.py: in a full environment (built ext,
triton) the real module is imported; otherwise conv1d.py is executed standalone against a stub
parent tree (ext / util.tensor / -- if the host genuinely lacks it -- triton), which is removed
from sys.modules again so no stub leaks into other test files. conv1d.py defines its Triton
kernels at MODULE scope (unlike gated_delta_rule.py), so the triton stub only needs a decorator
passthrough and the tl.constexpr annotation object; the stubbed kernels are never executed
because the Triton dispatch path itself is replaced by a recorder.
"""
import importlib.util
import inspect
import os
import sys
import types

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODNAME = "exllamav3.modules.gated_delta_net_fn.conv1d"


def _load_conv_module():
    try:
        from exllamav3.modules.gated_delta_net_fn import conv1d as conv
        return conv
    except Exception:
        pass  # no built ext / no triton on this machine: execute the real file against stubs
    made = []

    def stub(pkg_name, attrs = None):
        mod = types.ModuleType(pkg_name)
        for k, v in (attrs or {}).items():
            setattr(mod, k, v)
        sys.modules[pkg_name] = mod
        made.append(pkg_name)
        return mod

    try:
        # conv1d.py imports triton at module scope; stub it only if the host genuinely lacks it,
        # so a container with real triton keeps its own package
        try:
            import triton                                   # noqa: F401
        except ImportError:
            tl = stub("triton.language", {"constexpr": object})   # annotation slot only;
            tri = stub("triton", {                                # kernels never run here
                "jit": lambda fn = None, **kw: (fn if callable(fn) else (lambda f: f)),
                "next_power_of_2": lambda n: 1 << max(1, int(n)).bit_length(),
                "cdiv": lambda a, b: -(-a // b),
                "language": tl,
            })
            tri.language = tl
        stub("exllamav3"); stub("exllamav3.modules"); stub("exllamav3.modules.gated_delta_net_fn")
        stub("exllamav3.util")
        stub("exllamav3.util.tensor", {
            "get_for_device": lambda input_dict, key, device, default = None: input_dict.get(key, default),
            "buffered_arange": lambda n, device = None: torch.arange(n, dtype = torch.int64),
        })
        stub("exllamav3.ext", {"exllamav3_ext": types.SimpleNamespace(
            cuda_causal_conv1d_update = None)})   # monkeypatched by the native fixture
        spec = importlib.util.spec_from_file_location(
            MODNAME, os.path.join(ROOT, "exllamav3", "modules", "gated_delta_net_fn", "conv1d.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # relative imports hit the stub sys.modules entries
    finally:
        for pkg_name in reversed(made):
            sys.modules.pop(pkg_name, None)
        sys.modules.pop(MODNAME, None)  # keep the whole stub tree out of later test files
    return mod


conv = _load_conv_module()


# --------------------------------------------------------------------------- recording mocks

@pytest.fixture
def probe(monkeypatch):
    """Replace the hardware-probe seam; apply() installs either a constant answer or a
    {device: bool} map, and returns the list of devices the module asked about"""
    state = {"calls": [], "answers": False}

    def fake_device_lacks(device):
        state["calls"].append(device)
        a = state["answers"]
        return a.get(device, False) if isinstance(a, dict) else a

    def apply(answers):
        state["answers"] = answers
        return state["calls"]
    monkeypatch.setattr(conv, "_device_lacks_bf16_dot", fake_device_lacks)
    return apply


@pytest.fixture
def native(monkeypatch):
    """Record every native ext conv call and write a sentinel through the output argument,
    standing in for the pre-compiled kernel's in-place update"""
    calls = []

    def fake_native(x, conv_state, slots, weight, bias, out, activation, history):
        calls.append(dict(x = x, conv_state = conv_state, slots = slots, weight = weight,
                          bias = bias, out = out, activation = activation, history = history))
        out.fill_(0.25)

    monkeypatch.setattr(conv.ext, "cuda_causal_conv1d_update", fake_native)
    return calls


@pytest.fixture
def triton_path(monkeypatch):
    """Replace the module's Triton dispatch function with a recorder"""
    calls = []

    def fake_triton(x, conv_state, slots, weight, bias, transpose_output = False, history = False):
        calls.append(dict(x = x, conv_state = conv_state, slots = slots, weight = weight,
                          bias = bias, transpose_output = transpose_output, history = history))
        return "TRITON-OUT"

    monkeypatch.setattr(conv, "causal_conv1d_update_slotted_triton", fake_triton)
    return calls


@pytest.fixture
def hosttorch(monkeypatch):
    """torch stand-in for the module: allocations happen on the host (the tests pin WHICH path
    runs; tensors never go to CUDA); the requested shape/dtype/device is recorded verbatim"""
    class Proxy:
        def __init__(self):
            self.allocs = []

        def __getattr__(self, name):
            return getattr(torch, name)

        def _alloc(self, fn, shape, dtype, device):
            self.allocs.append((tuple(shape), dtype, str(device)))
            return fn(shape, dtype = dtype)

        def empty(self, shape, dtype = None, device = None):
            return self._alloc(torch.empty, shape, dtype, device)

        def zeros(self, shape, dtype = None, device = None):
            return self._alloc(torch.zeros, shape, dtype, device)

    proxy = Proxy()
    monkeypatch.setattr(conv, "torch", proxy)
    return proxy


# --------------------------------------------------------------------------- fake inputs

DEV0 = torch.device("cuda:0")
DEV1 = torch.device("cuda:1")
DIM, K, STATE_SIZE, BSZ = 8, 4, 4 + 64, 2   # GDN-style (dim, conv_kernel_size, window, bsz)


class DevTensor:
    """Tensor facade for dispatch pinning: only the attributes the gate and the recorders
    read exist -- shape, dtype, device, is_cuda."""

    def __init__(self, shape, dtype = torch.bfloat16, device = DEV0, is_cuda = True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self.is_cuda = is_cuda


def _inputs(T, device = DEV0, x_dtype = torch.bfloat16, state = torch.bfloat16,
             weight_dtype = torch.bfloat16, bias = torch.bfloat16, k = K,
             history = False, is_cuda = True):
    """kwargs for conv.causal_conv1d_update; state=None selects the dummy-state branch"""
    x = DevTensor((BSZ, DIM, T), x_dtype, device, is_cuda)
    conv_state = None if state is None else DevTensor((4, DIM, STATE_SIZE), state, device, is_cuda)
    weight = DevTensor((DIM, k), weight_dtype, device, is_cuda)
    b = None if bias is None else DevTensor((DIM,), bias, device, is_cuda)
    slots = DevTensor((BSZ,), torch.int32, device, is_cuda)
    return dict(mixed_qkv = x, conv_state = conv_state, recurrent_slots = slots,
                conv1d_weight = weight, conv1d_bias = b, history = history)


# --------------------------------------------------------------------------- threshold on every device

@pytest.mark.parametrize("T", [1, 8, 32])
def test_decode_sized_stays_native_without_probe_on_any_device(T, probe, native, triton_path,
                                                               hosttorch):
    # behavior identical to the pre-RDNA2-patch world: native at <=32 regardless of hardware,
    # and the probe is NEVER evaluated (no import, no device query on the token hot path)
    probed = probe(True)     # would route long shapes native too if consulted
    kw = _inputs(T)
    out = conv.causal_conv1d_update(**kw)
    assert probed == [], "MAX_CUDA_SEQLEN must short-circuit before the arch probe"
    assert len(native) == 1 and triton_path == []
    assert out is native[0]["out"]
    assert native[0]["slots"] is kw["recurrent_slots"], "slots pass through unchanged"


@pytest.mark.parametrize("T", [1, 32])
def test_decode_history_semantics_untouched(T, probe, native, hosttorch):
    probe(True)
    conv.causal_conv1d_update(**_inputs(T, history = True))
    assert native[0]["history"] is True
    assert native[0]["activation"] is True


# --------------------------------------------------------------------------- long prefill dispatch

@pytest.mark.parametrize("T", [33, 64, 257, 2048])
def test_long_prefill_goes_native_on_rdna2(T, probe, native, triton_path, hosttorch):
    probed = probe(True)     # gfx103x: Triton conv aborts; native has no length limit
    kw = _inputs(T)
    out = conv.causal_conv1d_update(**kw)
    assert probed == [kw["mixed_qkv"].device], "asked once, for the tensor's own device"
    assert triton_path == [], "the process-aborting Triton path must not be entered"
    call = native[0]
    assert call["x"] is kw["mixed_qkv"] and call["conv_state"] is kw["conv_state"]
    assert call["slots"] is kw["recurrent_slots"], "slot indirection identical to decode"
    assert call["activation"] is True and call["history"] is False
    assert call["out"].shape == (BSZ, T, DIM) and call["out"].dtype == torch.bfloat16, \
        "transposed (bsz, seqlen, dim) output semantics unchanged"
    assert hosttorch.allocs[-1][2] == "cuda:0", "the allocation request still names the input device"
    assert torch.equal(out, torch.full_like(out, 0.25)), "the kernel's buffer is what is returned"


@pytest.mark.parametrize("T", [33, 2048])
def test_long_prefill_keeps_legacy_triton_threshold_elsewhere(T, probe, native, triton_path,
                                                              hosttorch):
    # RDNA3 / RDNA4 / CDNA / NVIDIA: the probe answers False and 32 stays the rule
    probed = probe(False)
    kw = _inputs(T)
    out = conv.causal_conv1d_update(**kw)
    assert probed == [DEV0]
    assert native == [], "other hardware must not be dragged into the RDNA2 workaround"
    assert len(triton_path) == 1
    assert triton_path[0]["transpose_output"] is True
    assert triton_path[0]["slots"] is kw["recurrent_slots"]
    assert out == "TRITON-OUT"


def test_probe_is_per_device_not_process_wide(probe, native, triton_path, hosttorch):
    # two GPUs of DIFFERENT kinds: the V620 (cuda:0) must go native long, the RDNA3 card
    # (cuda:1) must stay on its working Triton path -- and each call asks its own device
    probed = probe({DEV0: True, DEV1: False})
    conv.causal_conv1d_update(**_inputs(33, device = DEV0))
    conv.causal_conv1d_update(**_inputs(33, device = DEV1))
    assert probed == [DEV0, DEV1], "the tensor's own device is queried, never current/device 0"
    assert len(native) == 1 and native[0]["x"].device == DEV0
    assert len(triton_path) == 1 and triton_path[0]["x"].device == DEV1


@pytest.mark.parametrize("T", [33, 512])
def test_long_rdna2_native_forwards_history(T, probe, native, hosttorch):
    probe(True)
    conv.causal_conv1d_update(**_inputs(T, history = True))
    assert len(native) == 1
    assert native[0]["history"] is True, "rewindable history writes run through the same " \
                                          "native kernel that already serves <=32"


# --------------------------------------------------------------------------- eligibility beats arch

@pytest.mark.parametrize("override, desc", [
    (dict(x_dtype = torch.float16), "fp16 activations"),
    (dict(x_dtype = torch.float32), "fp32 activations"),
    (dict(state = torch.float32), "fp32 conv_state"),
    (dict(weight_dtype = torch.float16), "fp16 weight"),
    (dict(bias = torch.float32), "fp32 bias"),
    (dict(k = 17), "conv kernel larger than MAX_CUDA_K"),
    (dict(is_cuda = False), "CPU tensors"),
])
def test_noneligible_geometry_falls_back_even_on_rdna2(override, desc, probe, native, triton_path,
                                                       hosttorch):
    probed = probe(True)
    kw = _inputs(33, **override)
    out = conv.causal_conv1d_update(**kw)
    assert native == [], f"{desc} must keep the Triton fallback"
    assert len(triton_path) == 1, desc
    assert out == "TRITON-OUT"
    assert probed == [], "non-eligible geometry is decided without any device probe"


@pytest.mark.parametrize("k, want_native", [(1, True), (4, True), (16, True), (17, False)])
def test_max_cuda_k_boundary_matches_the_cpp_constraint(k, want_native, probe, native,
                                                       triton_path, hosttorch):
    # gdn.cu: TORCH_CHECK(K <= CONV1D_MAX_K = 16); Python MAX_CUDA_K mirrors it at every length
    probe(True)
    conv.causal_conv1d_update(**_inputs(64, k = k))
    assert (len(native) == 1) is want_native
    assert (len(triton_path) == 1) is (not want_native)


# --------------------------------------------------------------------------- conv_state None branch

def test_dummy_state_long_rdna2_goes_native_with_null_slots(probe, native, triton_path, hosttorch):
    # stateless call: native accepts slots == None and the freshly created (bsz, dim, K) bf16
    # dummy state -- identical semantics to the <=32 path of today
    probe(True)
    conv.causal_conv1d_update(**_inputs(33, state = None))
    assert triton_path == []
    assert len(native) == 1 and native[0]["slots"] is None
    created = native[0]["conv_state"]
    assert created.shape == (BSZ, DIM, K) and created.dtype == torch.bfloat16


def test_dummy_state_elsewhere_materializes_identity_slots(probe, native, triton_path, hosttorch):
    probe(False)
    conv.causal_conv1d_update(**_inputs(33, state = None))
    assert native == [] and len(triton_path) == 1
    assert torch.equal(triton_path[0]["slots"], torch.arange(BSZ, dtype = torch.int64)), \
        "buffered_arange identity slots for the triton path, exactly as before"


# --------------------------------------------------------------------------- seam hygiene

def test_probe_seam_is_function_local_import():
    # the vendor import must stay OUT of module scope (fla's utils package queries the Triton
    # driver at import); conv1d may only reach it when a long prefill actually needs the answer
    src = inspect.getsource(conv._device_lacks_bf16_dot)
    assert "from ...vendor.fla.utils import device_lacks_bf16_dot" in src
    path = os.path.join(ROOT, "exllamav3", "modules", "gated_delta_net_fn", "conv1d.py")
    with open(path) as f:
        module_src = f.read()
    head = module_src.split("def ", 1)[0]
    assert "vendor.fla" not in head, "no module-scope vendor import: the <=32 hot path cannot " \
                                     "have acquired an import cost from this change"


def test_legacy_threshold_constants_unchanged():
    assert conv.MAX_CUDA_SEQLEN == 32
    assert conv.MAX_CUDA_K == 16
