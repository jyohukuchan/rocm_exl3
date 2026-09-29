"""
GPU numerical prerequisite tests for Qwen3.8 Flash Next (Qwen4Exp, EXL3 3.05bpw) on the
V620 gfx1030 pair (Phase 5 of doc/rdna2_port_plan.md).

Scope (bounded prerequisite; synthetic tensors only, no model weights, no generation):
  * GDN gate/beta scalar ops (ext.gated_delta_net_fused_op_2)      - decode-path pre-kernel
  * GDN fused recurrent rule (ext.cuda_recurrent_gated_delta_rule) - slots, carry, history,
    including long prefill through it (gfx103x runs prefill natively by default, 37a96d5)
  * vendored chunk prefill (exllamav3.vendor.fla.chunk_gated_delta_rule) through the real
    gated_delta_rule_fn dispatch - forced with EXL3_RDNA2_GDN_CHUNK on gfx103x, where it
    runs FP32-promoted - with recurrent-state carry / reset in the cache layout
  * causal_conv1d_update CUDA/Triton dispatch (seqlen 32/33 boundary), state carry,
    history write layout (the form GDNLayerState.rewind consumes)
  * GDNLayerState clear / stash / unstash / rewind mechanics at the real geometry
  * sigmoid-gated RMSNorm (the model's output_gate_type is "sigmoid"): the ext kernel path
    and the module's fp32 torch fallback, at the real (48 heads x 128 dim) norm shape
  * GatedResidual (Qwen4Exp hyper-connections): fused gr_mix (R<=32) and GEMM path (R>32),
    apply_ (ext.hc_apply without comb) and the final-mixer forward(), hc_mult=4 real
  * QSA indexer weight-free surfaces: token_mask semantics, select_indices vs its torch
    reference incl. the tiled top-k merge (SEL_TILE patched), paged-plane selection vs ref,
    qsa_sparse_attend_rows vs an fp64 gather-softmax, and the sparse_threshold() boundary
    (below the threshold the selection is provably dense, so sparse == causal attention)

Geometry from the actual model config
(https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3, branch 3.05bpw_h5_ng5; base Q
revision 69e33439ae950f17bcbe95c98f117d80f759ab6d):
  GDN  num_k_heads=16 num_v_heads=48 k/v_head_dim=128 conv_kernel=4 output_gate_type=sigmoid
  QSA  head_dim=256 num_q_heads=24 num_kv_heads=2 indexer n_heads=4 head_dim=128
       token_budget=2048 compress_ratio=4 (block_topk=512, sparse_threshold=2051)
  HC   hc_count=4 hc_lowrank=320 hidden=2560 rms_norm_eps=1e-6

References are independent PyTorch code written from the documented kernel semantics
(inspected gdn.cu, norm.cu, hc_mix.cu, the conv1d/gated_delta_rule dispatch and the
vendor/fla entry points): the v-head h <-> k-head h // (nv//nk) mapping, in-kernel l2norm
x*rsqrt(sum+1e-6), decay-before-readout delta rule, output *dk**-0.5, softplus linear
threshold 20, per-stream grouped RMSNorm with 1+w weight, low-rank silu->sigmoid mix gate
and 2*sigmoid inject gate. Nothing is asserted merely finite and no kernel is compared
only against itself.

Tolerances (justification per quantity, also noted inline):
  * bf16 kernel outputs: rtol 1.5e-2..3e-2, atol 1e-3..5e-3 - bf16 quantum is 2^-8 (3.9e-3
    relative); the recurrent kernel rounds toward zero, the chunk path additionally
    materialises bf16 l2normed q/k and chunk intermediates (on gfx103x it is FP32-promoted
    and measured at ~6e-5 vs a sequential reference, doc/qwen38_v620_results.md); recurrence
    drift over the short sequences tested stays under ~1%. NVIDIA-side equivalents
    (test_gated_delta_rule) passed at 5e-2, so these are tighter while insensitive only to
    sub-quantum noise.
  * fp32 states after <=130 steps: rtol/atol 2e-2 - accumulation-order and SFU exp/rsqrt
    differences only; a mapping or gating bug changes whole values (O(0.1..1) error).
  * half outputs (gated norm, gated residual): rtol 2e-3..2e-2 / atol 2e-4..6e-3 - half
    quantum 2^-11, plus fp16-rounded folded low-rank weights (gr path) or fp32 sum order.
  * pure copies / fp32 in-place updates: bit-exact, or rtol/atol 1e-6 where kernel and
    reference use identical arithmetic in a different association order.

Run on a GPU host (device cuda:0 by default, override with EXL3_TEST_DEVICE):
    python -m pytest -q tests/test_qwen38_rdna2.py
Skips cleanly on CPU-only hosts: exllamav3 (and the native extension) is imported lazily
inside CUDA-gated tests/fixtures, never at module or collection time.
Target: well under 5 minutes past first-time Triton autotune.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEV = os.environ.get("EXL3_TEST_DEVICE", "cuda:0")

# --------------------------------------------------------------- actual model geometry
GDN_NKH, GDN_NVK = 16, 48          # num_k_heads, num_v_heads (group = 3)
GDN_HK, GDN_HV = 128, 128          # linear_{key,value}_head_dim
CONV_K = 4                         # linear_conv_kernel_dim
FDIM_QKV = 2 * GDN_NKH * GDN_HK + GDN_NVK * GDN_HV      # 10240
HIDDEN = 2560
HC_MULT, HC_RANK = 4, 320          # hc_count, hc_lowrank
RMS_EPS = 1e-6
IDX_H, IDX_DK = 4, 128             # indexer_n_heads, indexer_head_dim
IDX_BUDGET, IDX_CR = 2048, 4       # -> block_topk = 512, sparse_threshold = 2051
ATTN_QH, ATTN_KVH, ATTN_HD = 24, 2, 256

_HAS_CUDA = torch.cuda.is_available() and DEV.startswith("cuda")
needs_cuda = pytest.mark.skipif(
    not _HAS_CUDA, reason="ROCm/CUDA GPU required (host runs are CPU-only)"
)

_DEV = torch.device(DEV) if _HAS_CUDA else None


def _device_ready():
    if not _HAS_CUDA:
        return False
    return _DEV.index is None or _DEV.index < torch.cuda.device_count()


# --------------------------------------------------------------- lazy production imports
@pytest.fixture(scope="module")
def ext():
    if not _device_ready():
        pytest.skip(f"device {DEV} unavailable")
    try:
        from exllamav3.ext import exllamav3_ext as m  # prebuilt in the GPU container
    except Exception as e:
        pytest.skip(f"native extension unavailable: {e}")
    return m


@pytest.fixture(scope="module")
def gdn_fns():
    """causal_conv1d_update + gated_delta_rule_fn: the exact functions GatedDeltaNet.forward
    drives on the torch (non-BC) path."""
    if not _device_ready():
        pytest.skip(f"device {DEV} unavailable")
    from exllamav3.modules.gated_delta_net_fn import causal_conv1d_update, gated_delta_rule_fn
    return causal_conv1d_update, gated_delta_rule_fn


def _mk_qsa(**over):
    from exllamav3.modules.qsa_indexer import QSAIndexer
    kw = dict(config=None, key="qsa", hidden_size=HIDDEN, n_heads=IDX_H, kv_heads=1,
              head_dim=IDX_DK, token_budget=IDX_BUDGET, compress_ratio=IDX_CR,
              rms_norm_eps=RMS_EPS)
    kw.update(over)
    # config=None -> NullConfig; the projection Linears are constructed but never loaded or
    # called: every method exercised below runs on explicit tensors only.
    return QSAIndexer(**kw)


def _mk_gr(D, rank, use_combine=True):
    from exllamav3.modules.hyperconnections import GatedResidual
    gr = GatedResidual(
        config=None, key="attn_hyper_connection", hc_mult=HC_MULT, hidden_size=D,
        rms_norm_eps=RMS_EPS, use_combine=use_combine,
    )
    gr.device = _DEV
    torch.manual_seed(31337 + D + rank)
    hs = HC_MULT * D
    gr.norm_w_raw = torch.randn(hs, dtype=torch.float32, device=_DEV) * 0.05
    down = torch.randn(rank, hs, dtype=torch.float32, device=_DEV) * 0.02
    up = torch.randn(hs, rank, dtype=torch.float32, device=_DEV) * 0.02
    inject = (torch.randn(HC_MULT, hs, dtype=torch.float32, device=_DEV) * 0.05
              if use_combine else None)
    gr._prepare(down, up, inject)
    return gr


# --------------------------------------------------------------- independent references
def _l2(x):
    # Convention inspected in gdn.cu (rsqrtf(sum + 1e-6)) and vendor/fla/l2norm.py
    # (1/sqrt(sum + eps), eps=1e-6): identical.
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def ref_gdn_recurrent(mixed_qkv, g, beta, state0, nk, nv, dk, dv, history_steps=False):
    """Vectorized fp32 gated-delta-rule recurrence per the documented semantics: q/k
    l2-normed, v head h uses k/q head h // group (group = nv//nk); per token
    s <- state*exp(g); u <- beta*(v - s^T k); state <- s + k (x) u;
    out <- q^T state * dk**-0.5.
    state0: (B, nv, dk, dv) fp32 carried-in state per batch row.
    Returns (out (B,T,nv,dv) fp32, final (B,nv,dk,dv) per batch, step states after tokens
    0..T-2 when history_steps - exactly the rows the CUDA kernel checkpoints for rewind)."""
    b, t, _ = mixed_qkv.shape
    group = nv // nk
    kdim, vdim = nk * dk, nv * dv
    q, k, v = torch.split(mixed_qkv.float(), [kdim, kdim, vdim], dim=-1)
    q = _l2(q.view(b, t, nk, dk)).repeat_interleave(group, dim=2)
    k = _l2(k.view(b, t, nk, dk)).repeat_interleave(group, dim=2)
    v = v.view(b, t, nv, dv)
    scale = dk ** -0.5
    state = state0.clone()
    outs, steps = [], []
    for i in range(t):
        decay = g[:, i].exp().unsqueeze(-1)                     # (b, nv, 1)
        s = state * decay.unsqueeze(-1)                         # (b, nv, dk, dv)
        pred = torch.einsum("bhk,bhkv->bhv", k[:, i], s)
        u = beta[:, i].float().unsqueeze(-1) * (v[:, i] - pred)
        state = s + k[:, i].unsqueeze(-1) * u.unsqueeze(-2)
        outs.append(torch.einsum("bhk,bhkv->bhv", q[:, i], state) * scale)
        if history_steps and i < t - 1:
            steps.append(state.clone())
    steps = torch.stack(steps, dim=1) if steps else None        # (b, T-1, nv, dk, dv)
    return torch.stack(outs, dim=1), state, steps


def gen_gdn_case(bsz, T, nk, nv, dk, dv, seed):
    """Decode/prefill inputs shaped exactly as the torch GDN path feeds them: mixed_qkv
    (b,T,2*nk*dk + nv*dv) bf16 [q|k|v], g (b,T,nv) fp32 log-decay, beta (b,T,nv) bf16."""
    torch.manual_seed(seed)
    qkv_dim = 2 * nk * dk + nv * dv
    mixed = (torch.randn((bsz, T, qkv_dim), dtype=torch.float32, device=_DEV) * 0.3).bfloat16()
    g = torch.randn((bsz, T, nv), dtype=torch.float32, device=_DEV) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((bsz, T, nv), dtype=torch.float32, device=_DEV)).bfloat16()
    return mixed, g, beta


def _check(a, b, rtol, atol, label):
    torch.testing.assert_close(
        a.float(), b.float(), rtol=rtol, atol=atol, msg=lambda m: f"{label}:\n{m}")


def ref_conv_update(x, conv_state, slots, weight, bias, history):
    """fp32 depthwise causal conv + silu with slotted windows, from the inspected kernel
    semantics: window = [state[slot, :, :K], x]; out t = silu(bias + sum_k w_k*win[t+1+k])
    rounded bf16 RN (the K-deep window slides starting ONE element into the state, i.e.
    state[0] is already outside the receptive field); non-history stores the last K window
    values at [0:K], history stores the tail-aligned min(state_size, K+S) values at the end
    of the row."""
    b, f, s = x.shape
    k = weight.shape[-1]
    ss = conv_state.shape[-1]
    new_state = conv_state.clone()
    outs = []
    for bi in range(b):
        slot = int(slots[bi])
        win = torch.cat([conv_state[slot, :, :k].float(), x[bi].float()], dim=-1)  # (f, k+s)
        y = F.conv1d(win.unsqueeze(0), weight.float().unsqueeze(1),
                     bias.float() if bias is not None else None, groups=f)
        # the kernels slide a K-deep window over [state, x] starting ONE element into the
        # state (F.conv1d yields k+s+1-k+1 = s+1 outputs; the first, fully-old-state, is
        # dropped) -> keep the trailing s
        outs.append(F.silu(y[0][:, -s:]).t().to(torch.bfloat16))                  # (s, f)
        total = k + s
        if history:
            write = min(ss, total)
            new_state[slot, :, ss - write:].copy_(win[:, total - write:].to(torch.bfloat16))
        else:
            new_state[slot, :, :k].copy_(win[:, -k:].to(torch.bfloat16))
    return torch.stack(outs, dim=0), new_state


def _sets_match(ref_sel, got_sel, score_of, topk, tie_tol):
    """Compare selected block sets allowing fp16 boundary ties (the kernel topks over an
    fp16 score slab; the reference scores are fp64). score_of maps block id -> fp64 score.
    Ties may only swap members AT the n-th-largest cut of the union; anything else fails."""
    if ref_sel == got_sel:
        return True
    if len(ref_sel) != len(got_sel):
        return False
    union = ref_sel | got_sel
    n = len(got_sel)
    if n == 0:
        return True
    desc = sorted((score_of[j] for j in union), reverse=True)
    boundary = desc[min(n, len(desc)) - 1]
    band = tie_tol * max(abs(boundary), 1e-3)
    return all(abs(score_of[j] - boundary) <= band for j in (ref_sel ^ got_sel))


# =============================================================== GDN scalar gate ops
@needs_cuda
@pytest.mark.parametrize("a_log_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("bsz,s", [(1, 1), (2, 7)])
@torch.inference_mode()
def test_gdn_fused_op2_beta_g(ext, bsz, s, a_log_dtype):
    """ext.gated_delta_net_fused_op_2: beta = sigmoid(b)*beta_scale -> bf16 RN, and
    g = -exp(a_log) * softplus(a + dt_bias) with the linear threshold at 20 (both inspected
    in gdn.cu). Real nv=48, fp32 b/a like the split-projection Linears, bf16 dt_bias/a_log.
    Checks the ROCm SFU __expf/log1pf path against exact fp64 math and the softplus tail."""
    torch.manual_seed(11 + s)
    nv = GDN_NVK
    b = torch.randn((bsz, s, nv), dtype=torch.float32, device=_DEV)
    a = torch.randn((bsz, s, nv), dtype=torch.float32, device=_DEV) * 2.0
    a[0, 0, :4] += 24.0          # well past the softplus linear threshold (20)
    a[0, 0, -4:] -= 30.0         # deep into the exp() tail (softplus ~ exp)
    dt_bias = (torch.randn((nv,), dtype=torch.float32, device=_DEV) * 0.5).bfloat16()
    a_log = (torch.randn((nv,), dtype=torch.float32, device=_DEV) * 0.6).to(a_log_dtype)

    beta_out = torch.empty((bsz, s, nv), dtype=torch.bfloat16, device=_DEV)
    g_out = torch.empty((bsz, s, nv), dtype=torch.float32, device=_DEV)
    ext.gated_delta_net_fused_op_2(b, a, dt_bias, a_log, beta_out, g_out, 1.0)

    d64 = torch.float64
    ref_beta = torch.sigmoid(b.to(d64)).bfloat16()
    ref_g = -torch.exp(a_log.float().to(d64)) * F.softplus(
        (a + dt_bias.float()).to(d64), beta=1.0, threshold=20.0)
    # beta: bf16 RN of a fast sigmoid vs exact - at most one bf16 ulp (2^-8 rel) boundary
    # flip; more than that means the SFU exp is broken on this GPU.
    _check(beta_out, ref_beta, rtol=8e-3, atol=1e-4, label="fused_op_2 beta")
    # g: expf/log1pf fp32 vs fp64 (~1e-5 rel); bound 5e-3 leaves SFU-quality headroom
    # while still failing a wrong threshold or sign.
    _check(g_out, ref_g, rtol=5e-3, atol=2e-3, label="fused_op_2 g")


# =============================================================== GDN recurrent rule
@needs_cuda
@pytest.mark.parametrize(
    "bsz,s,history,nk,nv,dk,dv",
    [
        (1, 1, False, GDN_NKH, GDN_NVK, GDN_HK, GDN_HV),   # real geometry, single decode
        (2, 3, False, GDN_NKH, GDN_NVK, GDN_HK, GDN_HV),   # real geometry, short chunk
        (1, 8, False, GDN_NKH, GDN_NVK, GDN_HK, GDN_HV),
        (2, 5, True, GDN_NKH, GDN_NVK, GDN_HK, GDN_HV),    # history checkpoint rows
        (1, 4, True, GDN_NKH, GDN_NVK, GDN_HK, GDN_HV),
        (2, 3, False, 2, 6, 64, 64),                       # group=3 small-dim path
        (2, 3, False, 2, 4, 128, 128),
    ],
)
@torch.inference_mode()
def test_gdn_recurrent_kernel_matches_reference(ext, bsz, s, history, nk, nv, dk, dv):
    """ext.cuda_recurrent_gated_delta_rule vs the independent fp32 recurrence, through the
    cache-shaped state tensor (slots, rows, nv, dk, dv) with a nontrivial slot map. Checks
    the v->k head mapping (h // group), l2norm/scale convention, decay-before-readout delta
    rule, final-state writeback to row 0, history checkpoint rows 1..s-1, slot isolation
    and bit-reproducibility across runs."""
    slots_idx = [2, 0] if bsz == 2 else [1]
    num_slots = 4
    rows = s if history else 1            # kernel writes rows 1..s-1 + final at row 0
    mixed, g, beta = gen_gdn_case(bsz, s, nk, nv, dk, dv, seed=4000 + s * 7 + bsz)
    torch.manual_seed(4001 + s)
    state = torch.randn((num_slots, rows, nv, dk, dv), dtype=torch.float32, device=_DEV) * 0.05
    state_before = state.clone()
    slots = torch.tensor(slots_idx[:bsz], dtype=torch.int32, device=_DEV)

    out = torch.empty((bsz, s, nv, dv), dtype=torch.bfloat16, device=_DEV)
    ext.cuda_recurrent_gated_delta_rule(mixed, g, beta, state, out, nk, nv, dk, dv,
                                        slots, history)
    out2 = torch.empty_like(out)
    state2 = state_before.clone()
    ext.cuda_recurrent_gated_delta_rule(mixed, g, beta, state2, out2, nk, nv, dk, dv,
                                        slots, history)
    # fixed-order reductions: identical inputs must give bit-identical outputs (shared
    # memory atomics on gfx1030 would surface here; NVIDIA-side equivalent exists)
    assert torch.equal(out, out2) and torch.equal(state, state2), \
        "recurrent kernel not bit-reproducible"

    ref0 = state_before[:, 0].index_select(0, slots.long())
    ref_out, ref_final, ref_steps = ref_gdn_recurrent(
        mixed, g, beta, ref0, nk, nv, dk, dv, history_steps=history)
    # out: bf16 round-toward-zero (__float2bfloat16_rz): <= 2^-8 of the value plus fp32
    # order noise; a mapping/gating bug changes values wholesale.
    _check(out, ref_out, rtol=2e-2, atol=2e-3, label="recurrent out")
    gathered = state.index_select(0, slots.long())
    _check(gathered[:, 0], ref_final, rtol=2e-2, atol=2e-3, label="recurrent state row 0")
    if history and s > 1:
        _check(gathered[:, 1:s], ref_steps, rtol=2e-2, atol=2e-3,
               label="recurrent history rows")

    # slot isolation: untouched slots stay bit-identical, and with history=False the
    # kernel must leave rows 1.. alone
    used = set(slots_idx[:bsz])
    for slot in range(num_slots):
        if slot not in used:
            assert torch.equal(state[slot], state_before[slot]), f"slot {slot} was touched"
    if not history:
        # history_stride == 1 in these cases; assert nothing beyond row 0 was written
        for slot in used:
            assert torch.equal(state[slot, 1:], state_before[slot, 1:])


@needs_cuda
@pytest.mark.parametrize("bsz", [1, 2])
@pytest.mark.parametrize("T", [47, 48, 130])
@torch.inference_mode()
def test_gdn_prefill_dispatch_branches_match_reference(gdn_fns, monkeypatch, bsz, T):
    """Long-prefill dispatch in gated_delta_rule_fn. The chunk branch is eligible when
    seqlen >= num_v_heads (real 48) and not history; since 37a96d5 gfx103x DEFAULTS long
    prefill to the fused native recurrent kernel (EXL3_RDNA2_GDN_CHUNK opts the vendored
    chunk kernels back in - they run FP32-promoted there, see vendor/fla). T=47 is native on
    every device. T=48/130 are checked BOTH ways: default dispatch first, then with the
    chunk branch forced, all against the same fp32 recurrence with state write-back into the
    cache slot (save_state=True). On NVIDIA the two runs are the same branch (cheap
    duplicate); on V620 the forced run is the real vendored-chunk numerical check."""
    _, gated_delta_rule_fn = gdn_fns
    nk, nv, dk, dv = GDN_NKH, GDN_NVK, GDN_HK, GDN_HV
    slots_idx = [1, 4] if bsz == 2 else [3]
    mixed, g, beta = gen_gdn_case(bsz, T, nk, nv, dk, dv, seed=5000 + T)
    torch.manual_seed(5001 + T)
    state = torch.zeros((6, 1, nv, dk, dv), dtype=torch.float32, device=_DEV)
    state.uniform_(-0.001, 0.001)               # tiny nonzero carried-in state
    state_before = state.clone()
    slots_dev = torch.tensor(slots_idx, dtype=torch.int32, device=_DEV)
    params = {"recurrent_slots": slots_dev.cpu()}

    ref0 = state_before[:, 0].index_select(0, slots_dev.long())
    ref_out, ref_final, _ = ref_gdn_recurrent(mixed, g, beta, ref0, nk, nv, dk, dv)

    runs = ["default"] + (["chunk"] if T >= nv else [])
    for n, mode in enumerate(runs):
        if n:                                      # second branch: fresh mutated state
            state.copy_(state_before)
        if mode == "chunk":
            monkeypatch.setenv("EXL3_RDNA2_GDN_CHUNK", "1")
        out = gated_delta_rule_fn(
            mixed_qkv=mixed, beta=beta, g=g, recurrent_state=state,
            recurrent_slots=slots_dev, history=False, save_state=True,
            num_k_heads=nk, num_v_heads=nv, k_dim=nk * dk, v_dim=nv * dv,
            k_head_dim=dk, v_head_dim=dv, params=params)
        assert out.shape == (bsz, T, nv, dv)
        # chunk path materialises bf16 l2normed q/k and chunk states on NVIDIA; on RDNA2 it
        # runs FP32-promoted (measured vs sequential ref: ~6e-5, doc/qwen38_v620_results.md).
        # Recurrent path is rz-rounded bf16. One bound covers all branches conservatively.
        _check(out, ref_out, rtol=3e-2, atol=5e-3, label=f"rule {mode} out T={T}")
        got = state.index_select(0, slots_dev.long())[:, 0]
        _check(got, ref_final, rtol=2e-2, atol=2e-3, label=f"rule {mode} state T={T}")
    used = set(slots_idx)
    for slot in range(6):
        if slot not in used:
            assert torch.equal(state[slot], state_before[slot]), f"slot {slot} touched"


@needs_cuda
@torch.inference_mode()
def test_gdn_state_carry_prefill_decode_reset(gdn_fns, monkeypatch):
    """The production sequence, twice: chunk-prefill 64 tokens into a cache slot
    (save_state=True writes the slot's row 0), then 4 fused-recurrent decode steps
    continuing from that slot. On gfx103x long prefill DEFAULTS to the native recurrent
    kernel (37a96d5), so the sequence first runs with default dispatch, then again after
    zeroing the slot with EXL3_RDNA2_GDN_CHUNK=1 forcing the vendored chunk kernels - that
    second phase is the chunk-state -> native-decode handoff through the cache row. Both
    segments must match one fp32 recurrence over the whole 68 tokens and the written cache
    row must equal the reference mid-state; after a reset (zero_ of the slot, what
    GDNLayerState.clear does) a fresh decode must match a from-zero reference; and
    save_state=False must leave the cache row untouched."""
    _, gated_delta_rule_fn = gdn_fns
    nk, nv, dk, dv = GDN_NKH, GDN_NVK, GDN_HK, GDN_HV
    slot = 3
    slots_dev = torch.tensor([slot], dtype=torch.int32, device=_DEV)
    params = {"recurrent_slots": slots_dev.cpu()}
    T1, T2 = 64, 4

    mixed_a, g_a, beta_a = gen_gdn_case(1, T1, nk, nv, dk, dv, seed=7001)
    mixed_b, g_b, beta_b = gen_gdn_case(1, T2, nk, nv, dk, dv, seed=7002)

    def run_rule(mixed, g, beta, save_state=True):
        return gated_delta_rule_fn(
            mixed_qkv=mixed, beta=beta, g=g, recurrent_state=state,
            recurrent_slots=slots_dev, history=False, save_state=save_state,
            num_k_heads=nk, num_v_heads=nv, k_dim=nk * dk, v_dim=nv * dv,
            k_head_dim=dk, v_head_dim=dv, params=params)

    state = torch.zeros((4, 1, nv, dk, dv), dtype=torch.float32, device=_DEV)
    out_a = run_rule(mixed_a, g_a, beta_a)          # long-prefill branch (64 >= 48;
    carried = state[slot, 0].clone()                # native by default on gfx103x)
    out_b = run_rule(mixed_b, g_b, beta_b)          # recurrent path (4 < 48)

    mixed = torch.cat([mixed_a, mixed_b], dim=1)
    g = torch.cat([g_a, g_b], dim=1)
    beta = torch.cat([beta_a, beta_b], dim=1)
    ref0 = torch.zeros((1, nv, dk, dv), dtype=torch.float32, device=_DEV)
    ref_out, ref_final, ref_steps = ref_gdn_recurrent(mixed, g, beta, ref0, nk, nv, dk, dv,
                                                      history_steps=True)
    _check(out_a, ref_out[:, :T1], rtol=3e-2, atol=5e-3, label="prefill segment")
    # ref_steps[i] = state after token i, i.e. after T1 tokens it is index T1-1
    _check(carried, ref_steps[0, T1 - 1], rtol=2e-2, atol=2e-3,
           label="carried state after prefill")
    _check(out_b, ref_out[:, T1:], rtol=2e-2, atol=2e-3, label="decode segment (carry)")
    _check(state[slot, 0], ref_final[0], rtol=2e-2, atol=2e-3, label="final state (carry)")

    # Second phase with the vendored chunk kernels FORCED for prefill (NVIDIA default; on
    # gfx103x this is the EXL3_RDNA2_GDN_CHUNK opt-in and the chunk runs FP32-promoted):
    # a chunk-written cache row feeding a native decode - the cross-backend handoff.
    state.zero_()
    monkeypatch.setenv("EXL3_RDNA2_GDN_CHUNK", "1")
    out_a2 = run_rule(mixed_a, g_a, beta_a)
    carried2 = state[slot, 0].clone()
    monkeypatch.delenv("EXL3_RDNA2_GDN_CHUNK", raising=False)
    out_b2 = run_rule(mixed_b, g_b, beta_b)
    _check(out_a2, ref_out[:, :T1], rtol=3e-2, atol=5e-3, label="prefill segment (chunk forced)")
    _check(carried2, ref_steps[0, T1 - 1], rtol=2e-2, atol=2e-3,
           label="carried state (chunk forced)")
    _check(out_b2, ref_out[:, T1:], rtol=2e-2, atol=2e-3, label="decode after chunk prefill")
    _check(state[slot, 0], ref_final[0], rtol=2e-2, atol=2e-3,
           label="final state (chunk forced carry)")

    # reset the slot and decode the same 4 tokens from scratch
    state[slot].zero_()
    out_reset = run_rule(mixed_b, g_b, beta_b)
    ref_r_out, ref_r_final, _ = ref_gdn_recurrent(mixed_b, g_b, beta_b, ref0, nk, nv, dk, dv)
    _check(out_reset, ref_r_out, rtol=2e-2, atol=2e-3, label="decode after reset")
    _check(state[slot, 0], ref_r_final[0], rtol=2e-2, atol=2e-3, label="state after reset")

    # save_state=False: chunk computes WITH the carried state but must not write the row
    state[slot, 0].copy_(ref_r_final[0] * 0.5)     # arbitrary marker != prefill result
    marker = state[slot, 0].clone()
    run_rule(mixed_a, g_a, beta_a, save_state=False)
    assert torch.equal(state[slot, 0], marker), "save_state=False mutated the cache row"


# =============================================================== causal conv1d update
@needs_cuda
@pytest.mark.parametrize("s,history", [(1, False), (8, False), (32, False), (33, False),
                                        (64, False), (8, True), (40, True)])
@torch.inference_mode()
def test_conv1d_update_paths_and_carry(gdn_fns, s, history):
    """causal_conv1d_update at the real fdim_qkv=10240 / K=4, slotted state, silu. Dispatch
    is inside the function: native CUDA kernel for s<=32 on every device; for s>32 native
    again on gfx103x (where the triton conv cannot compile at prefill sizes, 37a96d5) and
    triton elsewhere. Numerics are checked against the same fp32 reference regardless of
    branch; the bf16 state window is a pure copy -> bit-exact."""
    causal_conv1d_update, _ = gdn_fns
    bsz = 2
    slots_idx = [0, 2]
    ss = CONV_K + (8 if history else 0)
    f = FDIM_QKV
    torch.manual_seed(6000 + s)
    x = (torch.randn((bsz, f, s), dtype=torch.float32, device=_DEV) * 0.5).bfloat16()
    weight = (torch.randn((f, CONV_K), dtype=torch.float32, device=_DEV) * 0.5).bfloat16()
    bias = (torch.randn((f,), dtype=torch.float32, device=_DEV) * 0.25).bfloat16()
    state = (torch.randn((3, f, ss), dtype=torch.float32, device=_DEV) * 0.5).bfloat16()
    slots = torch.tensor(slots_idx, dtype=torch.int32, device=_DEV)
    state_before = state.clone()               # the kernels update the window IN PLACE

    out = causal_conv1d_update(x, state, slots, weight, bias, history=history, params={})
    assert out.shape == (bsz, s, f) and out.dtype == torch.bfloat16

    ref_out, ref_state = ref_conv_update(x, state_before, slots, weight, bias, history)
    # fp32 conv+silu chain, bf16 RN output: <= 1 ulp + sum-order noise
    _check(out, ref_out, rtol=1.6e-2, atol=3e-3, label=f"conv out s={s} hist={history}")
    # the state window is a pure copy of raw bf16 values -> bit-exact, incl. untouched slots
    assert torch.equal(state, ref_state), f"conv state window mismatch s={s} hist={history}"


@needs_cuda
@torch.inference_mode()
def test_conv1d_state_carry_two_chunks(gdn_fns):
    """Conv state carry: two 8-token calls through the CUDA kernel must reproduce one
    16-token call bit-exactly (the window is raw input; per-step fp32 arithmetic is the
    same sequence). A state-carry or window-positioning bug changes the second half."""
    causal_conv1d_update, _ = gdn_fns
    f = FDIM_QKV
    torch.manual_seed(6100)
    x = (torch.randn((1, f, 16), dtype=torch.float32, device=_DEV) * 0.5).bfloat16()
    weight = (torch.randn((f, CONV_K), dtype=torch.float32, device=_DEV) * 0.5).bfloat16()
    bias = (torch.randn((f,), dtype=torch.float32, device=_DEV) * 0.25).bfloat16()
    init = (torch.randn((1, f, CONV_K), dtype=torch.float32, device=_DEV) * 0.5).bfloat16()
    slots = torch.tensor([0], dtype=torch.int32, device=_DEV)

    state_whole = init.clone()
    out_whole = causal_conv1d_update(x, state_whole, slots, weight, bias, history=False,
                                     params={})
    state_split = init.clone()
    out_1 = causal_conv1d_update(x[:, :, :8].contiguous(), state_split, slots, weight, bias,
                                 history=False, params={})
    out_2 = causal_conv1d_update(x[:, :, 8:].contiguous(), state_split, slots, weight, bias,
                                 history=False, params={})
    assert torch.equal(out_whole[:, :8], out_1), "first chunk outputs differ"
    assert torch.equal(out_whole[:, 8:], out_2), "second chunk outputs differ (carry bug)"
    assert torch.equal(state_whole, state_split), "final window differs"


# =============================================================== gated RMSNorm
@needs_cuda
@pytest.mark.parametrize("gate_activation", ["sigmoid", "silu"])
@torch.inference_mode()
def test_gated_rmsnorm_real_norm_shape(gate_activation):
    """GatedRMSNorm at the GDN output norm shape (x bf16 (1,3,48,128), fp32 gate z, half
    out): Qwen3.8-Flash-Next sets output_gate_type="sigmoid", so the sigmoid variant IS the
    production one; silu covers the other consumers (Qwen3-Next-style GDN). The
    bf16-contiguous module path runs ext.gated_rms_norm (gate after the weighted norm);
    the sigmoid module path also runs its fp32 torch fallback - both checked against one
    fp64 reference: y = (x * rsqrt(mean(x^2)+eps) * w) * act(gate)."""
    from exllamav3.modules.gated_rmsnorm import GatedRMSNorm
    import torch.nn as nn
    shape = (1, 3, GDN_NVK, GDN_HV)
    torch.manual_seed(8001)
    x32 = torch.randn(shape, dtype=torch.float32, device=_DEV)
    w32 = (1.0 + torch.randn((GDN_HV,), dtype=torch.float32, device=_DEV) * 0.1)
    gate = torch.randn(shape, dtype=torch.float32, device=_DEV)

    mod = GatedRMSNorm(config=None, key="norm", rms_norm_eps=RMS_EPS, out_dtype=torch.half,
                       gate_activation=gate_activation)
    w_bf = w32.bfloat16()                        # as loaded via stc (allow_bf16)
    mod.weight = nn.Parameter(w_bf)

    d64 = torch.float64
    act = torch.sigmoid if gate_activation == "sigmoid" else F.silu

    def ref64(x):
        h = x.to(d64)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        return (h * w_bf.to(d64) * act(gate.to(d64))).to(torch.half)

    x_bf = x32.bfloat16()
    y_ext = mod.forward(x_bf, {}, gate=gate)     # bf16 contiguous -> ext kernel
    _check(y_ext, ref64(x_bf), rtol=2e-3, atol=2e-4,
           label=f"gated_rms_norm ext {gate_activation}")
    if gate_activation == "sigmoid":
        # fp32 x triggers the module fallback (different path, same documented formula)
        y_torch = mod.forward(x32, {}, gate=gate)
        _check(y_torch, ref64(x32), rtol=2e-3, atol=2e-4,
               label="gated_rms_norm torch fallback")
    assert y_ext.shape == shape and y_ext.dtype == torch.half


# =============================================================== gated residual
@needs_cuda
@pytest.mark.parametrize("D,rank,R", [
    (256, 64, 8),                    # small, fused gr_mix path
    (HIDDEN, HC_RANK, 8),            # real hidden/rank, fused path (decode-class R)
    (256, 64, 40),                   # GEMM path (prefill-class R)
    (HIDDEN, HC_RANK, 40),           # real hidden/rank, GEMM path
])
@torch.inference_mode()
def test_gated_residual_mix_and_apply(D, rank, R):
    """GatedResidual.mix (site form, hc_mult=4): low-rank gated stream mix + 2*sigmoid
    inject gates, vs an independent fp64 CPU reference built from the SAME half-rounded
    weights that _prepare hands the kernels:
    normed = x*rsqrt(mean+eps)*(1+w); t = silu(flat@down^T/H); w = sigmoid(t@up^T);
    mixed = mean_h(w * normed); post = 2*sigmoid(flat@inject^T/H).
    Then apply_ (ext.hc_apply with comb=None): x += post (x) y in place, fp32."""
    gr = _mk_gr(D, rank, use_combine=True)
    b = 1
    streams = torch.randn((b, R, HC_MULT, D), dtype=torch.float32, device=_DEV)

    post, comb, mixed = gr.mix(streams, {})
    assert comb is None
    post = post.clone(); mixed = mixed.clone()   # fused path returns static-cache views

    x64 = streams.double().cpu()
    w64 = gr.norm_w.double().cpu()               # (H, D), incl +1
    down64 = gr.down_h.float().double().cpu()    # weights as the kernels see them
    up64 = gr.up_h.float().double().cpu()
    inj64 = gr.inject_h.float().double().cpu()
    x_h = x64.reshape(R, HC_MULT, D)
    normed = x_h * torch.rsqrt(x_h.pow(2).mean(-1, keepdim=True) + RMS_EPS) * w64
    flat = normed.flatten(-2)
    t = F.silu(flat @ down64.t() / HC_MULT)
    gate = torch.sigmoid(t @ up64.t())
    ref_mixed = (gate.unflatten(-1, (HC_MULT, D)) * normed).mean(dim=-2)
    ref_post = 2.0 * torch.sigmoid(flat @ inj64.t() / HC_MULT)

    # mixed comes out half; the fused path folds (down*w)/(inject*w) into half weights and
    # the GEMM path additionally rounds normed and the gate GEMM to half -> the 2^-11-scale
    # weight/product rounding dominates, bounded well below a mapping bug (O(0.1)).
    _check(mixed.reshape(R, D).cpu(), ref_mixed.to(torch.half),
           rtol=2e-2, atol=6e-3, label=f"gr mixed R={R} D={D}")
    _check(post.reshape(R, HC_MULT).cpu(), ref_post,
           rtol=8e-3, atol=4e-3, label=f"gr post R={R} D={D}")

    x = streams.clone()
    y = mixed.view(b, R, D)
    p = post.view(b, R, HC_MULT)
    gr.apply_(x, y, p, None, {})
    ref_x = streams + p.unsqueeze(-1) * y.float().unsqueeze(-2)
    _check(x, ref_x, rtol=2e-6, atol=2e-6, label=f"gr apply_ R={R} D={D}")


@needs_cuda
@torch.inference_mode()
def test_gated_residual_final_mixer_forward():
    """Final-mixer form (use_combine=False, the HF hyper_connection_mixer): forward()
    collapses the stream stack with the same mix math but NO inject gate - this exercises
    the fused gr_mix variant with post=None at decode-class R."""
    D, rank, R = 512, 96, 16
    gr = _mk_gr(D, rank, use_combine=False)
    streams = torch.randn((1, R, HC_MULT, D), dtype=torch.float32, device=_DEV)
    mixed = gr.forward(streams, {}, out_dtype=torch.float32)
    assert mixed.shape == (1, R, D)

    x64 = streams.double().cpu()
    x_h = x64.reshape(R, HC_MULT, D)
    normed = x_h * torch.rsqrt(x_h.pow(2).mean(-1, keepdim=True) + RMS_EPS) \
        * gr.norm_w.double().cpu()
    t = F.silu(normed.flatten(-2) @ gr.down_h.float().double().cpu().t() / HC_MULT)
    gate = torch.sigmoid(t @ gr.up_h.float().double().cpu().t())
    ref = (gate.unflatten(-1, (HC_MULT, D)) * normed).mean(dim=-2)
    _check(mixed.reshape(R, D).cpu(), ref, rtol=1.5e-2, atol=4e-3,
           label="gr final mixer forward")


# =============================================================== QSA selection
def _ref_scores(q, pooled, scale):
    """Independent fp64 block scores: sum_h relu(q_h . k) * scale (documented formula)."""
    s = torch.einsum("bshd,bnd->bshn", q.double(), pooled.double())
    return F.relu(s).sum(dim=2) * scale


@needs_cuda
@pytest.mark.parametrize("bsz,seq,past_len", [(1, 13, 0), (2, 9, 6), (1, 7, 3)])
@torch.inference_mode()
def test_qsa_token_mask_semantics(bsz, seq, past_len):
    """token_mask (eager full-mask selection) vs an independent per-row fp64
    reconstruction of the documented rule: a query at position p may attend token kv iff
    kv <= p AND (kv is in the incomplete tail block [nbq*cr, p] OR kv's complete block
    j < nbq = (p+1)//cr is among the top block_topk visible-scoring blocks). Small
    geometry (budget=8 -> topk=2, cr=4) with past_len spanning block boundaries; the
    real-geometry selection parity is covered by the select_indices tests below."""
    cr, topk = 4, 8 // 4
    idx = _mk_qsa(token_budget=8, compress_ratio=cr)
    torch.manual_seed(9000 + seq)
    nb = (past_len + seq) // cr
    q = torch.randn((bsz, seq, IDX_H, IDX_DK), dtype=torch.float16, device=_DEV) * 0.3
    pooled = torch.randn((bsz, nb, IDX_DK), dtype=torch.float16, device=_DEV) * 0.3

    mask = idx.token_mask(q, pooled, past_len, past_len + seq)
    assert mask.shape == (bsz, seq, past_len + seq)

    scores = _ref_scores(q, pooled, idx.scale)              # (b, seq, nb) fp64
    for bi in range(bsz):
        for t in range(seq):
            p = past_len + t
            nbq = (p + 1) // cr
            row = mask[bi, t].cpu()
            # causality and the always-on incomplete tail
            assert not row[p + 1:].any(), f"future token visible b{bi} t{t}"
            if nbq * cr <= p:
                assert row[nbq * cr:p + 1].all(), "tail block missing"
            sel = set(int(x) // cr for x in row.nonzero().flatten().tolist()
                      if int(x) // cr < nbq)
            k = min(topk, nbq)
            if k > 0:
                order = sorted(range(nbq), key=lambda j: (-scores[bi, t, j].item(), j))
                ref_sel = set(order[:k])
            else:
                ref_sel = set()
            if sel != ref_sel:
                score_of = {j: scores[bi, t, j].item() for j in range(nb)}
                assert _sets_match(ref_sel, sel, score_of, topk, tie_tol=2 ** -11), \
                    f"token_mask selection mismatch b{bi} t{t} past {past_len}"


@needs_cuda
@pytest.mark.parametrize("bsz,seq,past_len,budget",
                         [(1, 12, 0, 8), (2, 8, 5, 8), (1, 16, 0, IDX_BUDGET)])
@torch.inference_mode()
def test_qsa_select_indices_vs_reference(monkeypatch, bsz, seq, past_len, budget):
    """Kernel selection chain (dsa_indexer_scores fp16 slab -> ext.dsa_topk -> pool expand)
    vs select_indices_ref (torch): per row, the emitted token lists must include the full
    causal tail, never a future token, respect the topk count, and match the reference's
    selected block set (up to fp16 boundary ties). Then the tiled multi-merge path
    (SEL_TILE forced to 2 -> several score tiles) must agree bit-for-bit with the
    single-tile run (shared total order: score desc, index asc)."""
    idx = _mk_qsa(token_budget=budget)
    cr = idx.compress_ratio
    torch.manual_seed(9100 + seq)
    nb = (past_len + seq) // cr
    q_idx = torch.randn((bsz, seq, IDX_H, IDX_DK), dtype=torch.float16, device=_DEV) * 0.3
    pooled = torch.randn((bsz, nb, IDX_DK), dtype=torch.float16, device=_DEV) * 0.3
    scores = _ref_scores(q_idx, pooled, idx.scale)

    got = idx.select_indices(q_idx, pooled, past_len, batch_stride=seq)
    ref = idx.select_indices_ref(q_idx, pooled, past_len, batch_stride=seq)
    assert got.shape == ref.shape == (bsz * seq, idx.k_pad())
    assert got.dtype == torch.int32

    topk = idx.block_topk
    for bi in range(bsz):
        for t in range(seq):
            p = past_len + t
            base = bi * seq + t
            g_row = [int(x) for x in (got[base][got[base] >= 0].cpu() - bi * seq)]
            r_row = [int(x) for x in (ref[base][ref[base] >= 0].cpu() - bi * seq)]
            assert g_row, f"empty selection row {base}"
            assert max(g_row) <= p, "causality violated"
            nbq = (p + 1) // cr
            assert set(range(nbq * cr, p + 1)) <= set(g_row), "tail block missing"
            g_blocks = {x // cr for x in g_row if x // cr < nbq}
            r_blocks = {x // cr for x in r_row if x // cr < nbq}
            assert len(g_blocks) <= min(topk, nbq)
            if g_blocks != r_blocks:
                score_of = {j: scores[bi, t, j].item() for j in range(max(nb, 1))}
                assert _sets_match(r_blocks, g_blocks, score_of, topk, tie_tol=2 ** -10), \
                    f"select_indices mismatch b{bi} t{t}"
            # every emitted token is in a visible complete block or in the tail
            assert all(x // cr < nbq or x >= nbq * cr for x in g_row)

    monkeypatch.setattr(type(idx), "SEL_TILE", 2, raising=True)
    got_tiled = idx.select_indices(q_idx, pooled, past_len, batch_stride=seq)
    assert torch.equal(got.sort(dim=-1).values, got_tiled.sort(dim=-1).values), \
        "tiled top-k merge changed the selected tokens (order is not contractual)"


@needs_cuda
@torch.inference_mode()
def test_qsa_select_indices_paged_vs_reference():
    """Cache-resident selection (select_indices_paged scores the pooled plane through a
    block table) vs select_indices_paged_ref. Plane geometry: raw page 8 tokens, pool page
    2 pools (page tokens = bpp * cr, as the cache lays them out), permuted block table, one
    sequence at cache position 4."""
    from types import SimpleNamespace
    idx = _mk_qsa(token_budget=12)                 # topk 3, nb 4 -> real selection pressure
    cr, dk = idx.compress_ratio, idx.head_dim
    page_sz, bpp, bsz, seq, pos0 = 8, 2, 1, 12, 4
    nb = (pos0 + seq) // cr
    pages = -(-nb // bpp) + 1                      # one spare page for the perm
    torch.manual_seed(9200)
    pooled_true = torch.randn((nb, dk), dtype=torch.float16, device=_DEV) * 0.3
    plane = torch.zeros((pages, bpp, dk), dtype=torch.float16, device=_DEV)
    bt = torch.randperm(pages, dtype=torch.int32).unsqueeze(0).contiguous()
    for j in range(nb):
        plane[bt[0, j // bpp], j % bpp] = pooled_true[j]
    bt = bt.to(_DEV)
    layer = SimpleNamespace(raw_k=torch.zeros((pages, page_sz, dk), dtype=torch.float16,
                                              device=_DEV),
                            pooled=plane)
    q_idx = torch.randn((bsz, seq, IDX_H, dk), dtype=torch.float16, device=_DEV) * 0.3
    seqlens = torch.tensor([pos0], dtype=torch.int32)
    scores = _ref_scores(q_idx, pooled_true.unsqueeze(0), idx.scale)[0]

    got = idx.select_indices_paged(layer, q_idx, bt, seqlens)
    ref = idx.select_indices_paged_ref(layer, q_idx, bt, seqlens)
    assert got.shape == ref.shape == (bsz * seq, idx.k_pad())

    topk = idx.block_topk
    for t in range(seq):
        p = pos0 + t
        nbq = (p + 1) // cr
        g = [int(x) for x in got[t][got[t] >= 0].cpu()]
        r = [int(x) for x in ref[t][ref[t] >= 0].cpu()]
        assert g and max(g) <= p, f"row {t}: empty or non-causal"
        assert set(range(nbq * cr, p + 1)) <= set(g), "tail missing (paged)"
        gb = {x // cr for x in g if x // cr < nbq}
        rb = {x // cr for x in r if x // cr < nbq}
        if gb != rb:
            assert _sets_match(rb, gb, {j: scores[t, j].item() for j in range(nb)},
                               topk, tie_tol=2 ** -10), f"paged selection mismatch row {t}"


# =============================================================== QSA sparse attention
@needs_cuda
@pytest.mark.parametrize("qh,kvh,hd", [(8, 2, 64), (ATTN_QH, ATTN_KVH, ATTN_HD)])
@torch.inference_mode()
def test_qsa_sparse_attend_rows(qh, kvh, hd):
    """qsa_sparse_attend_rows (gathered split-softmax GQA over per-row -1-padded token
    lists) vs an fp64 gather-softmax reference fed the SAME index lists (the selection
    itself is covered by the parity tests above). Checks the q->kv head mapping
    h // (qh//kvh) and the softmax/scale at the real 24/2/256 geometry."""
    from exllamav3.modules.attention_fn.qsa_triton import qsa_sparse_attend_rows
    torch.manual_seed(9300 + hd)
    R = 8
    q = torch.randn((R, qh, hd), dtype=torch.float16, device=_DEV) * 0.3
    k = torch.randn((R, kvh, hd), dtype=torch.float16, device=_DEV) * 0.3
    v = torch.randn((R, kvh, hd), dtype=torch.float16, device=_DEV) * 0.3
    group = qh // kvh
    sm_scale = hd ** -0.5
    K_pad = 8
    indices = torch.full((R, K_pad), -1, dtype=torch.int32, device=_DEV)
    for r in range(R):
        n = torch.randint(1, min(r + 1, K_pad) + 1, (1,)).item()
        sel = torch.randperm(r + 1)[:n]
        indices[r, :n] = sel.int()

    o = qsa_sparse_attend_rows(q, k, v, indices, sm_scale)
    assert o.shape == (R, qh, hd) and o.dtype == torch.float16

    q64, k64, v64 = q.double(), k.double(), v.double()
    for r in range(R):
        sel = indices[r][indices[r] >= 0].long()
        for h in range(qh):
            kvh_h = h // group
            logits = (q64[r, h] @ k64[sel, kvh_h].t()) * sm_scale
            ref = torch.softmax(logits, dim=-1) @ v64[sel, kvh_h]
            _check(o[r, h].unsqueeze(0), ref.unsqueeze(0), rtol=2e-2, atol=3e-3,
                   label=f"sparse attend r{r} h{h}")


@needs_cuda
@torch.inference_mode()
def test_qsa_sparse_equals_causal_below_threshold():
    """sparse_threshold() boundary at the real indexer config: block_topk=512 and
    threshold=4*512+3=2051 pins the arithmetic; for a 16-token prefill every query row has
    nb_q <= 4 visible blocks, so ALL causal tokens are selected and sparse_attend_nc must
    reproduce plain causal attention exactly (within fp16 noise)."""
    from types import SimpleNamespace
    idx = _mk_qsa()
    assert idx.block_topk == IDX_BUDGET // IDX_CR == 512
    assert idx.sparse_threshold() == 4 * idx.block_topk + 3 == 2051
    attn = SimpleNamespace(num_q_heads=ATTN_QH, num_kv_heads=ATTN_KVH, head_dim=ATTN_HD,
                           sm_scale=ATTN_HD ** -0.5)
    seq = 16
    assert seq < idx.sparse_threshold()
    torch.manual_seed(9400)
    b = 1
    q_idx = torch.randn((b, seq, IDX_H, IDX_DK), dtype=torch.float16, device=_DEV) * 0.3
    pooled = torch.randn((b, seq // IDX_CR, IDX_DK), dtype=torch.float16, device=_DEV) * 0.3
    q = torch.randn((b, seq, ATTN_QH, ATTN_HD), dtype=torch.float16, device=_DEV) * 0.3
    k = torch.randn((b, seq, ATTN_KVH, ATTN_HD), dtype=torch.float16, device=_DEV) * 0.3
    v = torch.randn((b, seq, ATTN_KVH, ATTN_HD), dtype=torch.float16, device=_DEV) * 0.3

    o = idx.sparse_attend_nc(attn, q, k, v, q_idx, pooled)

    group = ATTN_QH // ATTN_KVH
    scale = attn.sm_scale
    q64, k64, v64 = q.double(), k.double(), v.double()
    rows = []
    for t in range(seq):
        per_head = []
        for h in range(ATTN_QH):
            kvh_h = h // group
            logits = (q64[0, t, h] @ k64[0, :t + 1, kvh_h].t()) * scale
            per_head.append(torch.softmax(logits, dim=-1) @ v64[0, :t + 1, kvh_h])
        rows.append(torch.stack(per_head))
    o_ref = torch.stack(rows).view(b, seq, ATTN_QH, ATTN_HD)
    _check(o, o_ref, rtol=2e-2, atol=3e-3, label="sparse below threshold == causal")


# =============================================================== layer-state mechanics
@needs_cuda
@torch.inference_mode()
def test_gdn_layer_state_clear_stash_rewind(gdn_fns):
    """GDNLayerState at the real geometry (4 slots, max_history=8): alloc zeros both state
    tensors; clear zeroes a slot; stash/unstash round-trips the checkpointed [0:K] conv
    window and recurrent row 0 bit-exactly; and rewind after a history-recording chunk
    restores the exact prefix state (conv window from the tail-aligned history, recurrent
    state from row last_history+1-num_tokens). Validated against a parallel short run
    through the same kernels and the fp32 reference recurrence."""
    from exllamav3.modules.gated_delta_net import GDNLayerState
    from types import SimpleNamespace
    causal_conv1d_update, gated_delta_rule_fn = gdn_fns

    nk, nv, dk, dv = GDN_NKH, GDN_NVK, GDN_HK, GDN_HV
    mock = SimpleNamespace(fdim_qkv=FDIM_QKV, conv_kernel_size=CONV_K, num_v_heads=nv,
                           k_head_dim=dk, v_head_dim=dv)
    max_hist = 8
    ls = GDNLayerState(module=mock, max_batch_size=4, max_history=max_hist, cache_id=0)
    ls.alloc(_DEV)
    assert torch.count_nonzero(ls.conv_state) == 0
    assert torch.count_nonzero(ls.recurrent_state) == 0

    # stash/unstash round trip on slot 1
    torch.manual_seed(9500)
    ls.conv_state[1].copy_((torch.randn_like(ls.conv_state[1].float()) * 0.5).bfloat16())
    ls.recurrent_state[1].copy_(torch.randn_like(ls.recurrent_state[1]) * 0.05)
    snap_conv, snap_rec = ls.conv_state[1].clone(), ls.recurrent_state[1].clone()
    stashed = ls.stash(1)
    ls.clear(1)
    assert torch.count_nonzero(ls.conv_state[1]) == 0
    assert torch.count_nonzero(ls.recurrent_state[1]) == 0
    ls.unstash(1, stashed)
    assert torch.equal(ls.recurrent_state[1, 0], snap_rec[0])
    assert torch.equal(ls.conv_state[1, :, :CONV_K], snap_conv[:, :CONV_K])
    # unstash restores only the checkpointed parts; history rows stay cleared
    assert torch.count_nonzero(ls.recurrent_state[1, 1:]) == 0

    # rollback: T=6 tokens with history=True from a nonzero state on slot 2, rewind 3
    T, r = 6, 3
    ls.clear(2)
    init_conv = torch.randn((FDIM_QKV, CONV_K), dtype=torch.float32, device=_DEV) * 0.4
    init_rec = torch.randn((nv, dk, dv), dtype=torch.float32, device=_DEV) * 0.04
    ls.conv_state[2, :, :CONV_K].copy_(init_conv.bfloat16())
    ls.recurrent_state[2, 0].copy_(init_rec)
    conv_before, rec_before = ls.conv_state.clone(), ls.recurrent_state.clone()

    torch.manual_seed(9501)
    x_in = (torch.randn((1, FDIM_QKV, T), dtype=torch.float32, device=_DEV) * 0.4).bfloat16()
    w = (torch.randn((FDIM_QKV, CONV_K), dtype=torch.float32, device=_DEV) * 0.5).bfloat16()
    g = torch.randn((1, T, nv), dtype=torch.float32, device=_DEV) * 0.5 - 1.0
    beta = torch.sigmoid(torch.randn((1, T, nv), dtype=torch.float32, device=_DEV)).bfloat16()

    slots2 = torch.tensor([2], dtype=torch.int32, device=_DEV)
    conv_out = causal_conv1d_update(x_in, ls.conv_state, slots2, w, None, history=True,
                                    params={})
    gated_delta_rule_fn(
        mixed_qkv=conv_out, beta=beta, g=g, recurrent_state=ls.recurrent_state,
        recurrent_slots=slots2, history=True, save_state=True,
        num_k_heads=nk, num_v_heads=nv, k_dim=nk * dk, v_dim=nv * dv,
        k_head_dim=dk, v_head_dim=dv, params={})

    ls.rewind(2, T - 1, r)

    # parallel prefix run (first r tokens) from the same start on slot 3
    ls.clear(3)
    ls.conv_state[3, :, :CONV_K].copy_(init_conv.bfloat16())
    ls.recurrent_state[3, 0].copy_(init_rec)
    slots3 = torch.tensor([3], dtype=torch.int32, device=_DEV)
    conv_out_r = causal_conv1d_update(x_in[:, :, :r].contiguous(), ls.conv_state, slots3, w, None,
                                      history=False, params={})
    gated_delta_rule_fn(
        mixed_qkv=conv_out_r, beta=beta[:, :r].contiguous(), g=g[:, :r].contiguous(),
        recurrent_state=ls.recurrent_state, recurrent_slots=slots3, history=False,
        save_state=True, num_k_heads=nk, num_v_heads=nv, k_dim=nk * dk, v_dim=nv * dv,
        k_head_dim=dk, v_head_dim=dv, params={})

    # the rewound row-0 state is a bit copy of an in-kernel checkpoint: identical per-token
    # arithmetic to the short run, so only fp32 store/load noise could separate them
    _check(ls.recurrent_state[2, 0], ls.recurrent_state[3, 0], rtol=1e-6, atol=1e-6,
           label="rewind prefix state")
    assert torch.equal(ls.conv_state[2, :, :CONV_K], ls.conv_state[3, :, :CONV_K]), \
        "rewind conv window differs from the prefix run"

    # and the prefix state must be the reference state after r tokens
    ref_out, ref_final, _ = ref_gdn_recurrent(
        conv_out_r, g[:, :r], beta[:, :r], init_rec.unsqueeze(0), nk, nv, dk, dv)
    _check(ls.recurrent_state[2, 0], ref_final[0], rtol=2e-2, atol=2e-3,
           label="rewound state vs fp32 reference")

    # rewind must not touch other slots
    assert torch.equal(ls.conv_state[0], conv_before[0])
    assert torch.equal(ls.recurrent_state[0], rec_before[0])
