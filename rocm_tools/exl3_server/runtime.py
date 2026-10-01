#!/usr/bin/env python3
"""Runtime bring-up helper for exl3_server: the VERIFIED V620 TP2 + Qwen3.8 MTP
config (RCCL TP, unsharded MTP draft beside the output rank, quantized KV, Engram
single RAM owner + mlock) behind the same model_init argparse surface the
server already uses, plus lifecycle / audit / power-policy integration so HTTP
requests run on a proven placement instead of a hoped-for one.

Owner boundary: this file is the runtime worker's ONLY module. server.py is
root-owned and must not be edited here; everything below is written so root can
wire it in with a handful of calls (see INTEGRATION at the bottom of this
docstring and add_helper_flags() for the three new flags: --power-socket,
--context-limit, --max-output-tokens).

Paths
  load_runtime(args):
    * tensor_parallel AND mtp  -> verified specialized path, mirroring the
      reviewed rocm_tools.rdna2.tp_run load sequence exactly:
        - draft = Model.from_config(config, component="mtp"), UNSHARDED on
          tp_run.DRAFT_DEVICE ("cuda:1"), loaded BEFORE the target;
        - target = Model.load(use_per_device=<-gs>, tensor_p=True,
          tp_backend="nccl" (RCCL; tp_run documents 'native' as not ported, so
          the verified path uses the RCCL backend even though model_init's own
          -tpb default is 'native'), tp_output_device=tp_run.TP_OUTPUT_DEVICE
          ("cuda:1") so logits land next to the draft);
        - SAME quant cache kwargs for target and draft (K5/V4 when -cq 5,4),
          the target cache carrying max_history = resolved draft window
          (num_draft_tokens or the draft's default_draft_size cap; 4 for the
          Qwen3.8 MTP head), the draft cache deliberately without max_history
          (tp_run/model_init parity);
        - ngram_stream_from_disk=False from -ngr, and (because the verified
          config demands one physical CPU-RAM owner per Engram key with locked
          residency) EXL3_NGRAM_MLOCK=1 must already be in the environment at
          STARTUP. Nothing here ever touches RLIMIT_* or sysfs; when a limit is
          too small the error text names what the launcher must raise
          (ulimit -l / systemd LimitMEMLOCK) -- validation, not autoeditOS.
        - fail-closed audits reusing the public validated helpers:
          tp_run.tp_audit_rank / tp_cpu_helper_meta dispatched into every rank,
          tp_run.aggregate_tp_audit (arch, per-device placement, one Engram
          RAM owner per key, per-PID memory, pseudo output rank == parent),
          tp_run.attach_ngram_residency (non-faulting mincore page proof),
          tp_run.run_cache_runtime_audit (requested K/V bits vs ACTUAL cache
          layers, silent FP16 fallback rejected), parent-shell stray-table
          scan, plus this module's tp_mlock_rank dispatch for the mlock
          residency facts. Any problem unloads the models, drains only
          half-spawned rank contexts (tp_run._drain_tp_workers semantics:
          bounded joins, never the in-process pseudo rank, never os._exit)
          and raises RuntimeStartupError.
    * anything else            -> the generic path: exllamav3.model_init.init
          (which keeps its own MTP draft device selection -- different from the
          benchmark's, but unchanged for every non-verified model), followed by
          the same style of parent-side audit the LS branch of tp_run runs
          (multi_gpu.collect_ngram_state + mincore residency + one-owner
          validation + requested-vs-actual cache alignment).

Report
  runtime_report(audit) is pure stdlib dict-in/dict-out (no torch, no
  exllamav3, no /proc): /props embeds it so clients see the ACTUAL loaded_tp,
  backend (requested vs per-rank observed classes), devices/output rank, cache
  policy requested-vs-observed, MTP placement, Engram physical owners and
  locked page residency, never a re-statement of the CLI flags.

Power
  PowerContext attaches the reviewed rocm_tools.rdna2.power_policy adapter to
  the SYNC generator behind AsyncGenerator.generator (power_policy hooks
  iterate_start_jobs / iterate_draftmodel_*_gen / on_queue_drained, which exist
  on Generator, not on the async wrapper). batch == 1 therefore gets
  prefill=auto / profile_peak from the first draft+verify compute / auto on
  drain; batch > 1 gets held peak with auto restored on exit -- the exact
  validated policy split. For TP runs the adapter's torch is a facade whose
  cuda.synchronize(dev) dispatches tp_run.tp_sync_rank INTO the owning rank
  process (a parent-side torch.cuda.synchronize cannot drain a spawned TP
  context; the facade refuses unowned devices instead of lying about a drained
  queue). Cleanup order is fixed: close the AsyncGenerator first, restore the
  power policy (helper RPC + hooks + socket) BEFORE unloading draft/target and
  owned rank contexts, and never os._exit from library code.

INTEGRATION (server.py, root-owned; illustrative, do not paste here):

    from rocm_tools.exl3_server import runtime
    # ... parser: model_init.add_args(...) then runtime.add_helper_flags(parser)
    rt = runtime.load_runtime(args, log=print)
    state.model, state.config = rt.model, rt.config
    state.cache, state.tokenizer = rt.cache, rt.tokenizer
    state.draft_model, state.draft_cache = rt.draft_model, rt.draft_cache
    state.context_length = rt.context_length      # -cs, capped by --context-limit
    # in lifespan startup, AFTER AsyncGenerator(...) exists:
    rt.bind(agen)
    power = rt.power_context(agen)                # --power-socket; no-op when unset
    power.__enter__()
    # /props handler: payload["runtime"] = rt.report()
    # in lifespan shutdown:  await rt.shutdown()  # agen.close -> power -> unload

No side effects at import: torch / exllamav3 / rocm_tools.rdna2 are imported
lazily inside the functions that need them (multi_gpu/tp_run/power_policy are
themselves CPU-safe, but staying lazy keeps this module importable in any
environment and keeps tp_mlock_rank picklable for spawned ranks).
"""
from __future__ import annotations

import os
import resource
import types

__all__ = [
    "RuntimeStartupError",
    "PowerContextError",
    "load_runtime",
    "validate_runtime_args",
    "settings_for",
    "parse_cache_quant",
    "requested_cache_from_args",
    "aligned_context",
    "runtime_report",
    "tp_mlock_rank",
    "parent_mlock_records",
    "TPRankSyncTorch",
    "PowerContext",
    "Runtime",
    "LoadedRuntime",
    "add_helper_flags",
    "NGRAM_MLOCK_ENV",
]

NGRAM_MLOCK_ENV = "EXL3_NGRAM_MLOCK"
_PAGE_FALLBACK = 256


class RuntimeStartupError(RuntimeError):
    """Fail-closed startup: problems carry the human-readable reasons and
    requirements the memory/limit actions the launcher (root) must take."""

    def __init__(self, problems, requirements=()):
        problems = list(problems)
        requirements = list(requirements)
        msg = "runtime startup failed: " + "; ".join(problems) if problems else "runtime startup failed"
        if requirements:
            msg += "\nLauncher requirements (this helper never changes OS limits): " + "; ".join(requirements)
        super().__init__(msg)
        self.problems = problems
        self.requirements = requirements


class PowerContextError(RuntimeError):
    """The adapter cannot be attached honestly (missing sync generator, bad
    batch/devices). Fails loudly, never silently degrades."""


def _tp():
    """The reviewed TP harness module (CPU-safe; rocm_tools.rdna2.tp_run pulls
    only stdlib at module level). Centralized so tests can see one seam."""
    from rocm_tools.rdna2 import tp_run
    return tp_run


def _dev_index(value):
    """'cuda:1' / 1 / torch.device-like -> int index or None (tp_run parity)."""
    idx = getattr(value, "index", None)
    if isinstance(idx, int) and not isinstance(idx, bool):
        return idx
    s = str(value)
    tail = s.rsplit(":", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _expert_orders(args):
    path = getattr(args, "tp_expert_order", None)
    if not path:
        return {}
    from exllamav3.util.expert_placement import load_orders
    return load_orders(path)


def parse_cache_quant(spec):
    """'-cq 5,4' -> (5, 4); '8' -> (8, 8); None/'' -> None (FP16 caches).
    Mirrors model_init's split semantics and CacheLayer_quant's 2..8 assert,
    so a bad spec is a clear startup error instead of a late engine assert."""
    if spec is None or str(spec).strip() == "":
        return None
    parts = [int(bits) for bits in str(spec).split(",")]
    if len(parts) == 1:
        k_bits = v_bits = parts[0]
    elif len(parts) == 2:
        k_bits, v_bits = parts
    else:
        raise ValueError("-cq/--cache_quant takes kv_bits or k_bits,v_bits")
    for name, val in (("k_bits", k_bits), ("v_bits", v_bits)):
        if not 2 <= val <= 8:
            raise ValueError(f"{name}={val} outside 2..8 (CacheLayer_quant asserts the same range)")
    return (k_bits, v_bits)


def requested_cache_from_args(args):
    """Requested-KV-policy record in the exact shape tp_run's cache audit
    consumes: the requested quantization widths when -cq is given, FP16 otherwise."""
    try:
        bits = parse_cache_quant(getattr(args, "cache_quant", None))
    except ValueError as e:
        return {"policy": "invalid", "layer_type": None, "k_bits": None, "v_bits": None,
                "note": str(e)}
    if bits is None:
        return {"policy": "fp16", "layer_type": "CacheLayer_fp16", "k_bits": None, "v_bits": None,
                "note": "no -cq given: FP16 caches (the specialized TP+MTP HTTP path requires quantized caches)"}
    return {"policy": "quant", "layer_type": "CacheLayer_quant", "k_bits": bits[0],
            "v_bits": bits[1],
            "note": "K{0}/V{1} for target AND draft (same kwargs); QSA attention auto-maps to "
                    "CacheLayer_qsa_quant; GDN recurrent state is not KV and stays FP32/BF16"
                    .format(bits[0], bits[1])}


class _Unset:
    __slots__ = ()

    def __repr__(self):
        return "<default>"


_UNSENTINEL = _Unset()


def _tp_const(name, fallback):
    """Public tp_run constant with a CPU-safe fallback (only the verified path
    needs these, and tp_run imports stdlib-only)."""
    try:
        return getattr(_tp(), name)
    except Exception:
        return fallback


def _page():
    try:
        return int(_tp().PAGE)
    except Exception:
        return _PAGE_FALLBACK


def aligned_context(cache_max_tokens, context_limit):
    """Usable n_ctx: the cache capacity, page-rounded down, optionally capped by
    --context-limit (never raised above the cache)."""
    page = _page()
    if cache_max_tokens is None:
        return context_limit
    usable = int(cache_max_tokens) // page * page
    if context_limit is not None:
        usable = min(usable, int(context_limit) // page * page)
    return usable


def settings_for(args):
    """Pure derivation of every runtime setting from the REAL model_init /
    server argparse dests (nothing is inferred later from side channels)."""
    tp = bool(getattr(args, "tensor_parallel", False))
    mtp = bool(getattr(args, "mtp", False))
    try:
        bits = parse_cache_quant(getattr(args, "cache_quant", None))
    except ValueError:
        bits = None
    gs = getattr(args, "gpu_split", None)
    gpu_split_error = None
    split = None
    if gs not in (None, "auto"):
        try:
            split = [float(a) for a in str(gs).split(",")]
        except ValueError:
            gpu_split_error = f"-gs {gs!r} is not a comma-separated list of GB budgets"
    return {
        "path": "verified_tp_mtp" if (tp and mtp) else "generic_model_init",
        "model_dir": getattr(args, "model_dir", None),
        "tensor_parallel": tp,
        "mtp": mtp,
        "draft_model_dir": getattr(args, "draft_model_dir", None),
        "num_draft_tokens": getattr(args, "num_draft_tokens", None),
        "ngram_match_min": getattr(args, "ngram_match_min", 0) or 0,
        "dynamic_draft": bool(getattr(args, "dynamic_draft", False)),
        "draft_confidence": getattr(args, "draft_confidence", 0.4),
        "ngram_ram": bool(getattr(args, "ngram_ram", False)),
        "mlock_env": os.environ.get(NGRAM_MLOCK_ENV, "0") == "1",
        "cache_quant": bits,
        "cache_requested": requested_cache_from_args(args),
        "cache_compand_a": float(getattr(args, "cache_compand_a", 0.0) or 0.0),
        "cache_size": getattr(args, "cache_size", None),
        "cpu_cache_size": float(getattr(args, "cpu_cache_size", 0.0) or 0.0),
        "recurrent_cache_size": float(getattr(args, "recurrent_cache_size", 4.0) or 4.0),
        "gpu_split": split,
        "gpu_split_error": gpu_split_error,
        "chunk_size": getattr(args, "chunk_size", 4096),
        "batch_size": int(getattr(args, "autosplit_max_batch_size", 1) or 1),
        "tp_backend_arg": getattr(args, "tp_backend", "native"),
        # Verified config: RCCL ('nccl'); tp_run documents native as not ported.
        "tp_backend_effective": "nccl" if tp else getattr(args, "tp_backend", "native"),
        "tp_output_device": _tp_const("TP_OUTPUT_DEVICE", "cuda:1") if tp else None,
        "draft_device": _tp_const("DRAFT_DEVICE", "cuda:1") if (tp and mtp) else None,
        "swa_full": bool(getattr(args, "swa_full", False)),
        "layer_map": getattr(args, "layer_map", None),
        "load_verbose": bool(getattr(args, "load_verbose", False)),
        "power_socket": getattr(args, "power_socket", None),
        "context_limit": getattr(args, "context_limit", None),
        "max_output_tokens": getattr(args, "max_output_tokens", None),
        "expect_arch_env": os.environ.get("EXL3_TP_EXPECT_ARCH"),
    }


def _read_limits():
    """READ ONLY OS limit facts for the requirements text; this helper NEVER
    calls setrlimit (the launcher/root arranges limits, we only make a failure
    legible -- tp_run/mlock follow the same rule)."""
    out = {}
    for name, attr in (("memlock", "RLIMIT_MEMLOCK"), ("nofile", "RLIMIT_NOFILE")):
        try:
            out[name] = tuple(resource.getrlimit(getattr(resource, attr)))
        except Exception:
            out[name] = None
    return out


def _fmt_limit(v):
    return "unlimited" if v == getattr(resource, "RLIM_INFINITY", -1) else f"{v // 1024} KiB" if v >= 1024 else str(v)


def validate_runtime_args(args):
    """CPU-safe startup validation. Returns (problems, requirements): problems
    block the load, requirements describe what the launcher must provide
    (memory limits, helper socket, KV policy). No engine import, no GPU call,
    no limit mutation."""
    problems, requirements = [], []
    st = settings_for(args)

    if not st["model_dir"]:
        problems.append("-m/--model_dir is required")

    if st["mtp"] and st["draft_model_dir"]:
        problems.append("-mtp and -dm/--draft_model_dir are mutually exclusive (model_init asserts this at load time)")
    if st["mtp"] and st["ngram_match_min"]:
        problems.append("-mtp and -ngram/--ngram_match_min cannot combine (Generator asserts draft_model XOR ngram drafting)")

    if st["cache_requested"]["policy"] == "invalid":
        problems.append(st["cache_requested"]["note"])
    if st["cache_size"] is None:
        problems.append("-cs/--cache_size must be resolved before load_runtime "
                        "(server.py resolves its None default to the model max context first)")
    elif int(st["cache_size"]) < _page() or int(st["cache_size"]) % _page():
        problems.append(f"-cs {st['cache_size']} must be a positive multiple of {_page()} (paged cache)")

    if st["chunk_size"] is None or int(st["chunk_size"]) < _page() or int(st["chunk_size"]) % _page():
        problems.append(f"-chunk_size {st['chunk_size']} must be a positive multiple of {_page()}")
    if st["num_draft_tokens"] is not None and int(st["num_draft_tokens"]) < 1:
        problems.append("-ndt/--num_draft_tokens must be >= 1 when set")
    if st["batch_size"] < 1:
        problems.append("-ambs/--autosplit_max_batch_size must be >= 1")
    dc = st["draft_confidence"]
    if dc is not None and not (0 < float(dc) <= 1):
        problems.append("-dc/--draft_confidence must be in (0, 1]")

    if st["gpu_split_error"]:
        problems.append(st["gpu_split_error"])

    if getattr(args, "tp_expert_order", None):
        if not st["tensor_parallel"] or getattr(args, "tp_moe_tensor_split", False):
            problems.append("--tp-expert-order requires -tp without --tp_moe_tensor_split")

    if st["tensor_parallel"]:
        # model_init asserts these rules at load time; surface them as clear
        # startup problems for EVERY -tp run (verified or generic).
        for flag, attr in (("-mcl/--moe_cpu_offload", "moe_cpu_offload"),
                           ("-mcs/--moe_cpu_split", "moe_cpu_split"),
                           ("-dmcl/--draft_moe_cpu_layers", "draft_moe_cpu_layers")):
            if int(getattr(args, attr, 0) or 0):
                problems.append(f"{flag} requires layer-split mode and is rejected with -tp "
                                "(model_init asserts the same rule; validated TP runs are full GPU)")
    if st["path"] == "verified_tp_mtp":
        # Reviewed V620 TP2 + MTP + Engram placement. K5/V4 is the benchmark
        # reference; the cache kernels support independent K/V widths 2..8.
        if getattr(args, "override", None):
            problems.append("-or/--override tensor replacement is not supported on the verified "
                            "TP+MTP path (generic -tp/-dm models via model_init keep supporting it)")
        if st["gpu_split"] is None:
            problems.append("-tp -mtp needs an explicit two-device budget: -gs 28,28 (the verified "
                            "V620 pair setting); 'auto' cannot prove the RCCL split")
        elif len(st["gpu_split"]) != 2 or any(v <= 0 for v in st["gpu_split"]):
            problems.append(f"-gs {st['gpu_split']!r}: the verified TP2 load needs exactly two "
                            "finite positive per-device budgets")
        if st["cache_requested"]["policy"] != "invalid":
            if st["cache_quant"] is None:
                problems.append("verified TP2+MTP requires quantized caches: pass -cq k_bits,v_bits "
                                "with each width in 2..8 (benchmark reference: -cq 5,4)")
        if st["tp_backend_arg"] not in (None, "nccl"):
            requirements.append(f"-tpb {st['tp_backend_arg']!r} was requested but the verified TP "
                                "load uses the RCCL ('nccl') backend; native is not assumed ported "
                                "(see tp_run) -- the specialized path forces nccl")
    if st["ngram_ram"] and not st["mlock_env"]:
        problems.append(f"-ngr/--ngram_ram was requested, so every Engram table must end up with one "
                        f"locked CPU-RAM owner, but {NGRAM_MLOCK_ENV} != '1' in the environment. "
                        f"Export {NGRAM_MLOCK_ENV}=1 BEFORE starting the server (the lock is taken "
                        "while the tables load and the library fails explicitly rather than paging); "
                        "this helper will not raise RLIMIT_MEMLOCK for you.")

    if st["context_limit"] is not None:
        if int(st["context_limit"]) < _page():
            problems.append(f"--context-limit {st['context_limit']} is below one {_page()}-token page")
        elif st["cache_size"] is not None and int(st["context_limit"]) > int(st["cache_size"]):
            problems.append(f"--context-limit {st['context_limit']} exceeds -cs {st['cache_size']}: "
                            "a cap above the cache proves nothing and invites silent truncation")
    if st["max_output_tokens"] is not None and int(st["max_output_tokens"]) < 1:
        problems.append("--max-output-tokens must be >= 1")

    limits = _read_limits()
    if limits["memlock"] is not None:
        soft, hard = limits["memlock"]
        inf = getattr(resource, "RLIM_INFINITY", -1)
        if st["ngram_ram"] and soft != inf:
            requirements.append(
                f"RLIMIT_MEMLOCK soft={_fmt_limit(soft)} hard={_fmt_limit(hard)} (read here, never "
                f"modified): it must cover the whole Engram table the owner rank locks; the exact "
                "byte count appears in runtime_report under ngram after load. Raise it in the "
                "launcher (e.g. 'ulimit -l unlimited' before spawn or systemd LimitMEMLOCK).")
        if not st["ngram_ram"]:
            requirements.append(f"RLIMIT_MEMLOCK soft={_fmt_limit(soft)} hard={_fmt_limit(hard)} "
                                "(read-only note; only relevant with -ngr + EXL3_NGRAM_MLOCK=1)")
    if limits["nofile"] is not None:
        soft, _hard = limits["nofile"]
        if soft < 65536:
            requirements.append(f"RLIMIT_NOFILE soft={soft}: RCCL TP2 plus mmap'd safetensors were "
                                "validated at 65536; raise it in the launcher if fds run out "
                                "(load_runtime never calls setrlimit)")
    if st["power_socket"]:
        if not os.path.exists(str(st["power_socket"])):
            requirements.append(f"--power-socket {st['power_socket']!r} does not exist yet: start "
                                "the privileged power helper (root) before serving; PowerContext "
                                "attach fails loudly if it is still missing at generator startup")
        requirements.append("power policy: batch==1 runs auto during prefill, profile_peak from the "
                            "first draft+verify compute and auto again on queue drain; batch>1 holds "
                            "profile_peak for every request and restores auto on exit")
    else:
        requirements.append("no --power-socket given: requests run with whatever policy the "
                            "hardware is in; attach happens per AsyncGenerator when the socket is set")
    return problems, requirements


# ---------------------------------------------------------------------------
# Functions dispatched INTO TP ranks (top level here = picklable; the worker
# imports this module, which is CPU-safe).
# ---------------------------------------------------------------------------

def _mlock_records_for(holder):
    """Per-Engram-module mlock residency facts, read from the live objects
    (MlockedRanges owns the lock; properties are evaluated via getattr)."""
    from rocm_tools.rdna2 import multi_gpu
    recs = []
    for m in multi_gpu.find_ngram_modules(holder):
        lock = getattr(m, "_ram_lock", None)
        rec = {"key": str(getattr(m, "key", "?")), "has_lock_object": lock is not None,
               "is_locked": False, "locked_bytes": 0, "locked_ranges": 0}
        if lock is not None:
            try:
                rec["is_locked"] = bool(lock.is_locked)
                rec["locked_bytes"] = int(lock.locked_bytes)
                rec["locked_ranges"] = len(list(lock.locked_ranges))
            except Exception as e:
                rec["lock_error"] = repr(e)
        recs.append(rec)
    return recs


def tp_mlock_rank(local_context):
    """Run inside every rank: report whether each local n-gram table's pages
    are actually mlocked (EXL3_NGRAM_MLOCK=1) by THIS process. Observation
    only; the parent merges it into the audit and fails closed on a claimed
    RAM owner whose lock did not take."""
    devices = local_context.get("device")
    return {"device": devices, "pid": os.getpid(),
            "env_mlock": os.environ.get(NGRAM_MLOCK_ENV, "0"),
            "tables": _mlock_records_for(types.SimpleNamespace(
                modules=list(local_context.get("modules") or [])))}


def parent_mlock_records(model):
    """Layer-split / generic equivalent: the parent process IS the owner."""
    return {"device": None, "pid": os.getpid(),
            "env_mlock": os.environ.get(NGRAM_MLOCK_ENV, "0"),
            "tables": _mlock_records_for(model)}


# ---------------------------------------------------------------------------
# Power integration
# ---------------------------------------------------------------------------

class TPRankSyncTorch:
    """Torch facade for power_policy on TP runs: its ONLY torch use is
    cuda.synchronize(dev) at phase boundaries, and a parent synchronize cannot
    drain a spawned rank's context. Each call becomes an in-rank
    tp_run.tp_sync_rank dispatch; unowned devices are refused instead of
    reporting an un-drained boundary as policy-controlled."""

    def __init__(self, model):
        self.model = model
        self.syncs = []
        self.cuda = types.SimpleNamespace(synchronize=self.synchronize)

    def synchronize(self, dev):
        idx = _dev_index(dev)
        active = []
        for d in (getattr(self.model, "active_devices", None) or []):
            i = _dev_index(d)
            if i is not None:
                active.append(i)
        if idx is None or idx not in active:
            raise PowerContextError(
                f"cannot route power sync for device {dev!r} to a TP rank (active_devices {active}); "
                "refusing to label an unsynced boundary as policy-controlled")
        worker = self.model.tp_worker_dispatch_single(idx, _tp().tp_sync_rank, ())
        self.syncs.append({"device": idx, "worker": worker})
        return worker


class PowerContext:
    """Lifecycle adapter binding power_policy to the SYNC generator behind an
    AsyncGenerator. Enter AFTER the AsyncGenerator exists (its iteration task
    must be alive for hook timing), exit BEFORE any model unload so the helper
    restores 'auto' while the GPUs still own their queues. socket_path None
    keeps the context a recorded no-op (power policy is opt-in, exactly like
    power_policy itself)."""

    def __init__(self, async_generator, *, batch_size, devices, socket_path,
                 tp_model=None, torch_module=None):
        if socket_path is None:
            self.enabled = False
            self.socket_path = None
        else:
            self.enabled = True
            self.socket_path = str(socket_path)
        sync_gen = getattr(async_generator, "generator", None) if async_generator is not None else None
        if sync_gen is None or sync_gen is async_generator:
            # Attaching to the async wrapper would install hooks the sync
            # Generator never calls. Fail closed like power_policy does.
            if self.enabled:
                raise PowerContextError(
                    "PowerContext needs an AsyncGenerator (it attaches to its .generator, the sync "
                    "Generator whose iterate_* methods power_policy hooks); got "
                    f"{type(async_generator).__name__}")
            sync_gen = async_generator
        self.target = sync_gen
        self.batch = int(batch_size)
        self.devices = list(devices)
        if self.batch < 1:
            raise PowerContextError(f"batch_size must be >= 1, got {batch_size!r}")
        if self.enabled and (not self.devices or len(set(self.devices)) != len(self.devices)):
            raise PowerContextError(f"devices must be nonempty and unique, got {self.devices}")
        if tp_model is not None:
            self.torch = TPRankSyncTorch(tp_model)
        elif torch_module is not None:
            self.torch = torch_module
        elif self.enabled:
            import torch
            self.torch = torch
        else:
            self.torch = None
        self._policy = None
        self._active = False

    def __enter__(self):
        if not self.enabled:
            return self
        from rocm_tools.rdna2 import power_policy
        self._policy = power_policy.attach(self.target, self.torch, self.batch,
                                           self.devices, self.socket_path)
        self._policy.__enter__()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._active and self._policy is not None:
            try:
                return self._policy.__exit__(exc_type, exc, tb)
            finally:
                self._active = False
        return False

    @property
    def rank_syncs(self):
        return list(self.torch.syncs) if isinstance(self.torch, TPRankSyncTorch) else []

    def summary(self):
        if not self.enabled:
            return {"policy": "disabled", "note": "no --power-socket given; requests are NOT "
                    "labelled policy-controlled"}
        if self._policy is None:
            return {"policy": "pending", "note": "PowerContext created but never entered"}
        try:
            return self._policy.summary()
        except Exception as e:
            return {"policy": "unknown", "error": repr(e),
                    "note": "helper state unverified after this point"}


# ---------------------------------------------------------------------------
# The runtime object
# ---------------------------------------------------------------------------

class Runtime:
    """Everything server.py needs after a successful load. Plain attributes
    mirror ServerState so the wiring is assignment, not adaptation."""

    def __init__(self, *, model, config, cache, tokenizer, draft_model, draft_config,
                 draft_cache, settings, audit, log=None):
        self.model = model
        self.config = config
        self.cache = cache
        self.tokenizer = tokenizer
        self.draft_model = draft_model
        self.draft_config = draft_config
        self.draft_cache = draft_cache
        self.settings = dict(settings)
        self.audit = audit
        self.log = log or (lambda msg: None)
        self.cleanup_errors = []
        self._agen = None
        self._power = None
        self._peak_active_jobs = 0
        self._observed_generator = None
        self._original_iterate = None
        self._observed_iterate = None
        self._models_unloaded = False

    @property
    def is_tp(self):
        return bool(getattr(self.model, "loaded_tp", False)) \
            or self.settings["path"] == "verified_tp_mtp"

    @property
    def context_length(self):
        return aligned_context(getattr(self.cache, "max_num_tokens", None),
                               self.settings.get("context_limit"))

    @property
    def power_devices(self):
        return list(self.settings.get("power_devices") or [0])

    @property
    def generator_kwargs(self):
        """AsyncGenerator kwargs matching the loaded runtime and benchmark policy.
        num_draft_tokens is the RESOLVED draft depth (verified path: -ndt or the
        draft's default_draft_size), which never exceeds the cache's max_history
        window; on the generic path None lets the Generator resolve the draft's
        own default, exactly like model_init parity."""
        a = self.settings
        nd = a.get("draft_depth") or a.get("num_draft_tokens")
        return {
            "model": self.model,
            "cache": self.cache,
            "tokenizer": self.tokenizer,
            "draft_model": self.draft_model,
            "draft_cache": self.draft_cache,
            "num_draft_tokens": int(nd) if nd else None,
            "ngram_match_min": int(a.get("ngram_match_min") or 0),
            "dynamic_draft_tokens": bool(a.get("dynamic_draft", False)),
            "draft_confidence": float(a.get("draft_confidence", 0.4)),
            "max_batch_size": int(a.get("batch_size", 1)),
            "max_chunk_size": int(a.get("chunk_size", 2048)),
            "cpu_cache_size": int(float(a.get("cpu_cache_size", 0.0) or 0.0) * 1024**3),
            "recurrent_cache_size": int(float(a.get("recurrent_cache_size", 4.0) or 4.0) * 1024**3),
            "record_draft_stats": True,
        }

    def report(self):
        """/props payload: the static runtime_report over the stored audit plus
        the live power-policy state and generator concurrency, when bound."""
        report = runtime_report(self.audit)
        if self._power is not None:
            report["power_policy"] = self._power.summary()
            report["power_rank_syncs"] = self._power.rank_syncs
        sync_gen = getattr(self._agen, "generator", None) if self._agen is not None else None
        if sync_gen is not None:
            calibrator = getattr(sync_gen, "draft_calibrator", None)
            if calibrator is not None:
                report["draft_confidence"] = {
                    "invalid_estimates": getattr(calibrator, "invalid_estimates", 0),
                    "skipped_nonfinite_labels": getattr(calibrator, "skipped_nonfinite_labels", 0),
                }
            active_count = self._observe_active_jobs(sync_gen)
            current_peak = self._peak_active_jobs
            report["generator_runtime"] = {
                "active_jobs": active_count,
                "peak_active_jobs": current_peak,
                "configured_max_batch_size": getattr(sync_gen, "max_batch_size", None),
            }
        return report

    def bind(self, async_generator):
        """Remember the live AsyncGenerator so shutdown() closes it first."""
        self._restore_iteration_observer()
        self._agen = async_generator
        sync_gen = getattr(async_generator, "generator", None)
        iterate = getattr(sync_gen, "iterate", None) if sync_gen is not None else None
        if sync_gen is not None and iterate is not None:
            runtime = self
            from functools import wraps

            @wraps(iterate)
            def observed_iterate(*args, **kwargs):
                runtime._observe_active_jobs(sync_gen)
                try:
                    return iterate(*args, **kwargs)
                finally:
                    runtime._observe_active_jobs(sync_gen)

            sync_gen.iterate = observed_iterate
            self._observed_generator = sync_gen
            self._original_iterate = iterate
            self._observed_iterate = observed_iterate
        return async_generator

    def _observe_active_jobs(self, sync_gen):
        try:
            active = getattr(sync_gen, "active_jobs", None)
            count = len(active) if active is not None else None
        except Exception:
            return None
        if count is not None:
            self._peak_active_jobs = max(self._peak_active_jobs, count)
        return count

    def _restore_iteration_observer(self):
        sync_gen = self._observed_generator
        observed = self._observed_iterate
        if sync_gen is not None and observed is not None:
            try:
                if getattr(sync_gen, "iterate", None) is observed:
                    sync_gen.iterate = self._original_iterate
            finally:
                self._observed_generator = None
                self._original_iterate = None
                self._observed_iterate = None

    def power_context(self, async_generator, *, socket_path=_UNSENTINEL, batch_size=None,
                      devices=None):
        """Build (and remember) the PowerContext for this runtime. Defaults:
        socket from --power-socket, batch from -ambs (the cache slot count that
        defines the validated batch1/batch>1 policy split), devices from the
        audited placement. TP contexts get the in-rank tp_sync_rank facade."""
        sp = self.settings.get("power_socket") if socket_path is _UNSENTINEL else socket_path
        ctx = PowerContext(
            async_generator,
            batch_size=batch_size or self.settings.get("batch_size", 1),
            devices=devices or self.power_devices,
            socket_path=sp,
            tp_model=self.model if (self.is_tp and sp is not None) else None,
        )
        self._agen = async_generator if self._agen is None else self._agen
        self._power = ctx
        return ctx

    def unload_models(self):
        """Sync teardown: draft first, then target (whose unload() owns the
        loaded TP ranks); a HALF-spawned TP context (load failed before
        loaded_tp was set) is drained with tp_run's bounded join/terminate
        rules. Never os._exit; never touches processes we did not spawn."""
        if self._models_unloaded:
            return
        self._models_unloaded = True
        tp = _tp()
        for name, obj in (("draft", self.draft_model), ("model", self.model)):
            if obj is None:
                continue
            try:
                obj.unload()
            except Exception as e:
                self.cleanup_errors.append(f"unload {name}: {e!r}")
        try:
            if self.model is not None and not getattr(self.model, "loaded_tp", False):
                tp._drain_tp_workers(self.model, self.cleanup_errors)
        except Exception as e:
            self.cleanup_errors.append(f"tp worker drain: {e!r}")

    async def shutdown(self, async_generator=None, power=None):
        """Ordered cleanup: close the AsyncGenerator (its iteration task and
        job queues) -> restore the power policy (sync + helper 'auto' + hook
        removal) while models are still resident -> unload draft/target and
        owned rank contexts. Returns the collected cleanup errors."""
        agen = async_generator or self._agen
        ctx = power or self._power
        if agen is not None:
            try:
                close = getattr(agen, "close", None)
                if close is not None:
                    await close()
            except Exception as e:
                self.cleanup_errors.append(f"async generator close: {e!r}")
            finally:
                self._restore_iteration_observer()
        if ctx is not None:
            try:
                ctx.__exit__(None, None, None)
            except Exception as e:
                self.cleanup_errors.append(f"power restore: {e!r}")
        self.unload_models()
        return list(self.cleanup_errors)


LoadedRuntime = Runtime


# ---------------------------------------------------------------------------
# Load paths
# ---------------------------------------------------------------------------

def load_runtime(args, *, log=None):
    """Validate, load, audit and wrap the serving runtime. Raises
    RuntimeStartupError with actionable problems/requirements; on a mid-load
    audit failure every partially built resource is torn down before raising.
    'log' is an optional callable(str) for progress lines (server passes print)."""
    log = log or (lambda msg: None)
    problems, requirements = validate_runtime_args(args)
    if problems:
        raise RuntimeStartupError(problems, requirements)
    settings = settings_for(args)
    if settings["path"] == "verified_tp_mtp":
        return _load_verified_tp(args, settings, log, requirements)
    return _load_generic(args, settings, log, requirements)


def _load_verified_tp(args, settings, log, requirements):
    """Specialized verified load: model_init CANNOT express it (its MTP draft
    device selection differs from the benchmark and it never passes
    tp_output_device), so this mirrors tp_run.run()'s sequence and audits."""
    import torch
    from exllamav3 import Model, Config, Cache, Tokenizer
    from exllamav3.cache import CacheLayer_fp16, CacheLayer_quant
    from rocm_tools.rdna2 import multi_gpu, tp_run

    model = draft = None
    try:
        expect_arch = settings["expect_arch_env"] or tp_run.EXPECT_ARCH
        if torch.cuda.device_count() != 2 or any(
                torch.cuda.get_device_properties(d).gcnArchName.split(":")[0] != expect_arch
                for d in (0, 1)):
            raise RuntimeStartupError(
                [f"the verified TP+MTP load requires exactly 2 {expect_arch} GPUs (the V620 pair); "
                 f"device_count={torch.cuda.device_count()}"], requirements)

        config = Config.from_directory(args.model_dir, layer_map=settings["layer_map"])
        if settings["ngram_ram"]:
            config.infer_params.ngram_stream_from_disk = False
        has_mtp = "mtp" in (getattr(config, "model_classes", None) or {})
        if not has_mtp:
            raise RuntimeStartupError(
                [f"-mtp requires an 'mtp' component in {args.model_dir}; this checkpoint has none "
                 "(dense Qwen3 / MoE D/M models: drop -mtp and serve AR, or -dm a standalone draft, "
                 "which goes through the generic model_init path)"], requirements)

        model = Model.from_config(config, swa_full=settings["swa_full"])
        expected_ngram_keys = sorted({str(getattr(m, "key", None) or type(m).__name__)
                                      for m in multi_gpu.find_ngram_modules(model)})
        if expected_ngram_keys and not settings["ngram_ram"]:
            raise RuntimeStartupError(
                [f"this checkpoint exposes Engram table(s) {expected_ngram_keys}; the verified "
                 "placement requires -ngr (stream_from_disk=False) plus one locked RAM owner "
                 f"({NGRAM_MLOCK_ENV}=1) -- disk-streamed Engram must not pass"], requirements)
        if settings["ngram_ram"] and not expected_ngram_keys:
            log(" -- -ngr given but this TP+MTP checkpoint exposes no n-gram table (flag is a no-op)")

        draft = Model.from_config(config, swa_full=settings["swa_full"], component="mtp")

        # Resolved draft window: -ndt wins, else the draft's own default_draft_size
        # cap (4 for Qwen3.8 MTP); mirrors model_init's max_history formula.
        caps = getattr(draft, "caps", None) or {}
        default_ds = caps.get("default_draft_size") or 0
        ndt = settings["num_draft_tokens"] or 0
        # draft_window: cache max_history (>= any depth; tp_run parity floor of 4).
        # draft_depth: the ACTUAL Generator draft length (-ndt wins, else caps, else 4).
        draft_window = max(int(default_ds), int(ndt)) or 4
        draft_depth = int(ndt) if ndt else (int(default_ds) or 4)
        settings["draft_window"] = draft_window
        settings["draft_depth"] = draft_depth

        quant = settings["cache_quant"]
        if quant is None:
            cache_kwargs = {"layer_type": CacheLayer_fp16}
            expected_cls = CacheLayer_fp16
        else:
            cache_kwargs = {"layer_type": CacheLayer_quant, "k_bits": quant[0], "v_bits": quant[1],
                            "compand_a": settings["cache_compand_a"]}
            expected_cls = CacheLayer_quant

        cache = Cache(model, max_num_tokens=settings["cache_size"],
                      max_batch_size=settings["batch_size"], max_history=draft_window,
                      **cache_kwargs)
        # SAME quant cache kwargs for the draft; max_history stays target-side only
        # (tp_run/model_init parity), cache_size/batch identical so the Generator
        # draft-cache assertions hold.
        draft_cache = Cache(draft, max_num_tokens=settings["cache_size"],
                            max_batch_size=settings["batch_size"], **cache_kwargs)
        for cname, c in (("target", cache), ("draft", draft_cache)):
            got = getattr(c, "layer_type", None)
            if got is not expected_cls:
                raise RuntimeStartupError(
                    [f"{cname} Cache constructed layer_type {getattr(got, '__name__', got)!r} "
                     f"although {expected_cls.__name__} was requested -- no silent fallback"],
                    requirements)

        # Draft FIRST, unsharded, beside the future output rank (tp_run parity:
        # this is also what keeps AR-shaped loads device-layout identical).
        draft_dev = settings["draft_device"] or "cuda:1"
        log(f" -- Loading MTP draft unsharded on {draft_dev} (before the TP target)")
        draft.load(device=draft_dev, max_chunk_size=settings["chunk_size"],
                   progressbar=False, verbose=settings["load_verbose"])
        bad = [str(getattr(m, "device", None)) for m in (getattr(draft, "modules", None) or [])
               if _dev_index(getattr(m, "device", None)) != _dev_index(draft_dev)]
        if bad:
            raise RuntimeStartupError(
                [f"draft modules landed on {bad}, expected every one on {draft_dev}: the verified "
                 "placement keeps the MTP head UNSHARDED next to the output rank"], requirements)

        log(f" -- Loading target TP2 (backend=nccl/RCCL, output={settings['tp_output_device']})")
        tp_dev_limits = {}
        for key, arg_name in (("attn", "tp_max_parallelism_attn"), ("mlp", "tp_max_parallelism_mlp"),
                              ("moe", "tp_max_parallelism_moe"), ("linear", "tp_max_parallelism_linear"),
                              ("linear_attn", "tp_max_parallelism_linear_attn")):
            value = getattr(args, arg_name, None)
            if value is not None:
                tp_dev_limits[key] = value
        model.load(use_per_device=settings["gpu_split"], max_chunk_size=settings["chunk_size"],
                   progressbar=False, tensor_p=True, tp_backend="nccl",
                   tp_output_device=settings["tp_output_device"],
                   tp_dev_limits=tp_dev_limits,
                   tp_options={"moe_tensor_split": bool(getattr(args, "tp_moe_tensor_split", False)),
                               "expert_order": _expert_orders(args)},
                   verbose=settings["load_verbose"])

        loaded_tp = bool(getattr(model, "loaded_tp", False))
        actual_backend = getattr(model, "tp_backend", None)
        out_idx = _dev_index(getattr(model, "output_device", None))
        active = sorted({i for i in (_dev_index(d) for d in (getattr(model, "active_devices", None) or []))
                         if i is not None})
        problems = []
        if not loaded_tp:
            problems.append("TP load finished without model.loaded_tp -- the server would serve a "
                            "placement it did not ask for")
        if out_idx != _dev_index(tp_run.TP_OUTPUT_DEVICE):
            problems.append(f"model.output_device is {out_idx}, expected cuda:1: TP logits must "
                            "land next to the unsharded MTP draft")
        if active != [0, 1]:
            problems.append(f"model.active_devices resolved to {active}, expected exactly [0, 1] "
                            "-- a single-rank or duplicated layout must never be labelled TP2")
        if actual_backend not in (None, "nccl"):
            problems.append(f"model.tp_backend is {actual_backend!r}, requested 'nccl' (RCCL)")
        if problems:
            raise RuntimeStartupError(problems, requirements)

        rank_records = model.tp_worker_dispatch_wait_multi(active, tp_run.tp_audit_rank, ())
        mlock_records = model.tp_worker_dispatch_wait_multi(active, tp_mlock_rank, ())
        try:
            cpu_meta = model.tp_worker_dispatch_wait_multi([-1], tp_run.tp_cpu_helper_meta, ())
            cpu_meta = cpu_meta[0] if cpu_meta else None
        except Exception as e:
            cpu_meta = {"absent": True, "error": repr(e)}

        audit = tp_run.aggregate_tp_audit(rank_records, active, out_idx, os.getpid(),
                                          cpu_meta=cpu_meta, expect_arch=expect_arch,
                                          expected_ngram_keys=expected_ngram_keys)
        shell_recs = [multi_gpu.describe_ngram_module(m)
                      for m in multi_gpu.find_ngram_modules(model)]
        stray = [srec for srec in shell_recs if srec["num_table_tensors"] or srec["num_disk_handles"]]
        if stray:
            audit["problems"].append(
                f"parent TP shells still hold n-gram table(s)/handles: "
                f"{[(srec['key'], srec['mode'], srec['num_table_tensors'], srec['num_disk_handles']) for srec in stray]}")
            audit["ok"] = False
        _merge_mlock(audit, mlock_records, expected_ngram_keys, requirements)

        cache_probs, cache_notes, cache_obs = tp_run.run_cache_runtime_audit(
            settings["cache_requested"],
            tp_run._cache_geometries_from_modules(rank_records),
            tp_run._cache_geometries_from_model(draft))
        audit["cache"] = {"requested": settings["cache_requested"], "observed": cache_obs,
                          "notes": cache_notes, "problems": cache_probs}
        if cache_probs:
            audit["problems"].extend(cache_probs)
            audit["ok"] = False

        audit["path"] = settings["path"]
        audit["execution_actual"] = "tp2"
        audit["loaded_tp"] = loaded_tp
        audit["requested_backend"] = "nccl"
        audit["backend_actual"] = actual_backend
        audit["mtp"] = {
            "component_present": has_mtp,
            "device": draft_dev,
            "unsharded": True,
            "loaded_before_target": True,
            "same_cache_kwargs_as_target": True,
            "draft_depth": draft_depth,
            "draft_window": draft_window,
            "target_max_history": draft_window,
            "num_draft_tokens_arg": settings["num_draft_tokens"],
        }
        audit["power_devices"] = active
        settings["power_devices"] = active

        if not audit["ok"]:
            raise RuntimeStartupError(audit["problems"], requirements)

        tok = Tokenizer.from_config(config)
        if getattr(args, "load_metrics", False) and getattr(config, "stc", None) \
                and getattr(config.stc, "metrics", None):
            config.stc.metrics.print()
        return Runtime(model=model, config=config, cache=cache, tokenizer=tok,
                       draft_model=draft, draft_config=config, draft_cache=draft_cache,
                       settings=settings, audit=audit, log=log)
    except BaseException:
        _teardown_after_failed_load(model, draft)
        raise


def _teardown_after_failed_load(model, draft):
    """A failed verified load must not leak rank contexts: unload() owns them
    once loaded_tp was set; otherwise drain this process's children with
    tp_run's bounded rules (never the pseudo rank, never os._exit)."""
    try:
        tp_run = _tp()
    except Exception:
        return
    for obj in (draft, model):
        if obj is None:
            continue
        try:
            obj.unload()
        except Exception:
            pass
    if model is not None:
        try:
            if not getattr(model, "loaded_tp", False):
                tp_run._drain_tp_workers(model, [])
        except Exception:
            pass


def _merge_mlock(audit, mlock_records, expected_keys, requirements):
    """Attach per-rank mlock residency to the aggregated audit; a claimed RAM
    owner whose tables are NOT locked fails closed (verified config: single
    owner AND locked residency)."""
    by_dev = {r.get("device"): r for r in (mlock_records or []) if isinstance(r, dict)}
    tables, total_locked = [], 0
    for rec in mlock_records or []:
        for t in (rec.get("tables") or []):
            item = dict(t)
            item["device"] = rec.get("device")
            item["pid"] = rec.get("pid")
            tables.append(item)
            if t.get("is_locked"):
                total_locked += int(t.get("locked_bytes") or 0)
    audit["ngram_mlock"] = {"env_requested": bool(os.environ.get(NGRAM_MLOCK_ENV, "0") == "1"),
                            "records": by_dev, "tables": tables,
                            "locked_total_bytes": total_locked}
    for key, found in (audit.get("ngram_ram_owners") or {}).items():
        if not found:
            continue
        owner = found[0]
        lock = next((t for t in tables
                     if t.get("key") == key and t.get("device") == owner.get("device")
                     and t.get("pid") == owner.get("pid")), None)
        if lock is None or not lock.get("is_locked"):
            audit["problems"].append(
                f"Engram owner {key!r} (device {owner.get('device')}, PID {owner.get('pid')}) has "
                f"no ACTIVE {NGRAM_MLOCK_ENV} lock (record: {lock!r}): the RAM owner exists but its "
                "pages can still be paged out -- the verified single-owner-locked policy fails "
                "closed. Ensure EXL3_NGRAM_MLOCK=1 was exported before spawn and the owner "
                "process' RLIMIT_MEMLOCK covers the table.")
            audit["ok"] = False
    if expected_keys and not tables:
        audit["problems"].append(
            f"expected Engram key(s) {list(expected_keys)} but no rank reported any lock record: "
            f"locked residency unproven (was {NGRAM_MLOCK_ENV}=1 set before the tables loaded?)")
        audit["ok"] = False


def _load_generic(args, settings, log, requirements):
    """Any non-verified model: exllamav3.model_init.init, untouched, including
    its own MTP draft device selection (which may differ from the benchmark --
    that is fine for generic serving). Then the same style of audit tp_run runs,
    over whatever the load actually produced: rank dispatch for -tp loads
    (model_init sets loaded_tp but never passes tp_output_device, so no cuda:1
    demand here), parent-side walks for layer-split/single-device."""
    from exllamav3 import model_init
    from rocm_tools.rdna2 import multi_gpu, tp_run

    result = model_init.init(args)
    if len(result) == 7:
        model, config, cache, tokenizer, draft, draft_config, draft_cache = result
    else:
        (model, config, cache, tokenizer), draft, draft_config, draft_cache = result[:4], None, None, None

    loaded_tp = bool(getattr(model, "loaded_tp", False))
    devs = sorted({_dev_index(getattr(m, "device", None)) for m in (getattr(model, "modules", None) or [])} - {None})
    active = sorted({i for i in (_dev_index(d) for d in (getattr(model, "active_devices", None) or []))
                     if i is not None}) if loaded_tp else []
    out_idx = _dev_index(getattr(model, "output_device", None))
    settings["power_devices"] = (active or devs or [0])
    audit = {
        "path": "generic_model_init",
        "execution_actual": ("tp2" if loaded_tp and len(active) > 1 else
                             "layer_split" if len(devs) > 1 else
                             "tp" if loaded_tp else "single_device"),
        "loaded_tp": loaded_tp,
        "expected_devices": active or devs,
        "output_device": out_idx,
        "requested_backend": settings["tp_backend_arg"],
        "backend_actual": getattr(model, "tp_backend", None),
        "mtp": {"component_present": draft is not None, "requested": settings["mtp"],
                "device": None, "unsharded": None,
                "note": "generic model_init placement (draft device selection is model_init's, "
                        "which may differ from the verified benchmark)"},
        "problems": [], "notes": [], "ranks": [],
        "ngram_expected_keys": [], "ngram_ram_owners": {}, "ngram_replicated_copies": [],
        "memory": None, "cpu_helper": None,
    }
    cache_geoms = []
    if loaded_tp:
        try:
            rank_records = model.tp_worker_dispatch_wait_multi(active, tp_run.tp_audit_rank, ())
            mlock_records = model.tp_worker_dispatch_wait_multi(active, tp_mlock_rank, ())
            observed_keys = sorted({str(ng.get("key")) for r in rank_records
                                    for ng in ((r.get("ngram") or {}).get("modules") or [])})
            # model_init already loaded the parent shells, so a PRE-load Engram
            # key scan is impossible here; expected keys are what the ranks
            # observed, and -ngr still demands one locked RAM owner per key.
            expected = observed_keys if settings["ngram_ram"] else []
            # Generic path must not demand gfx1030; require only a HOMOGENEOUS
            # fleet by holding every rank to the first rank's arch base.
            obs_arch = str((rank_records[0].get("gcnArchName") if rank_records else "")
                           or "").split(":", 1)[0] or tp_run.EXPECT_ARCH
            merged = tp_run.aggregate_tp_audit(rank_records, active, out_idx, os.getpid(),
                                               expect_arch=obs_arch,
                                               expected_ngram_keys=expected)
            audit.update({k: merged[k] for k in
                          ("ok", "problems", "notes", "ranks", "memory",
                           "ngram_ram_owners", "ngram_replicated_copies", "arch_expected",
                           "parent_pid") if k in merged})
            audit["ngram_expected_keys"] = observed_keys
            _merge_mlock(audit, mlock_records, expected, requirements)
            cache_geoms = tp_run._cache_geometries_from_modules(rank_records)
        except Exception as e:
            audit["problems"].append(f"generic TP audit dispatch failed: {e!r}")
    else:
        if settings["ngram_ram"]:
            holder_modules = multi_gpu.find_ngram_modules(model)
            expected_keys = sorted({str(getattr(m, "key", "?")) for m in holder_modules})
            ng = multi_gpu.collect_ngram_state(model, require_ram=True)
            tp_run.attach_ngram_residency(ng.get("modules") or [], holder_modules)
            probs, owners, replicated = tp_run.validate_ngram_ownership(
                [(None, os.getpid(), rec) for rec in (ng.get("modules") or [])], expected_keys)
            audit["ngram_expected_keys"] = expected_keys
            audit["ngram_ram_owners"] = owners
            audit["ngram_replicated_copies"] = replicated
            audit["notes"].extend(ng.get("notes") or [])
            if probs:
                audit["problems"].extend(probs)
            if ng.get("problems"):
                audit["problems"].extend(ng["problems"])
            _merge_mlock(audit, [parent_mlock_records(model)], expected_keys, requirements)
        if len(devs) > 1:
            audit["placement"] = multi_gpu.audit_placement(model, devs)
            if not audit["placement"].get("ok"):
                audit["problems"].extend(audit["placement"].get("problems") or ["placement audit failed"])
        cache_geoms = tp_run._cache_geometries_from_model(model) if cache is not None else []
    if cache is not None:
        probs, notes, obs = tp_run.validate_cache_alignment(
            cache_geoms, settings["cache_requested"], label="generic target", require_kv=False)
        dprobs, dnotes = ([], [])
        if draft is not None:
            dprobs, dnotes, dobs = tp_run.validate_cache_alignment(
                tp_run._cache_geometries_from_model(draft), settings["cache_requested"],
                label="generic draft", require_kv=False)
            obs = {"target": obs, "draft": dobs}
        else:
            obs = {"target": obs}
        audit["cache"] = {"requested": settings["cache_requested"], "observed": obs,
                          "notes": list(notes) + list(dnotes), "problems": probs + dprobs}
        if probs or dprobs:
            audit["problems"].extend(probs + dprobs)
    audit["ok"] = not audit["problems"]
    if not audit["ok"]:
        for obj in (draft, model):
            try:
                if obj is not None:
                    obj.unload()
            except Exception as e:
                log(f" -- cleanup after failed generic audit: {e!r}")
        raise RuntimeStartupError(audit["problems"], requirements)
    return Runtime(model=model, config=config, cache=cache, tokenizer=tokenizer,
                   draft_model=draft, draft_config=draft_config, draft_cache=draft_cache,
                   settings=settings, audit=audit, log=log)


# ---------------------------------------------------------------------------
# /props report (pure CPU)
# ---------------------------------------------------------------------------

def runtime_report(audit):
    """Client-facing runtime facts derived from the stored audit dict. Pure
    function over already-collected data (no engine, no /proc, no GPU): safe to
    call per /props request. 'available' is False until a runtime has been
    loaded; every section degrades to None/empty on absent paths (generic
    single-device loads have no TP rank records, tableless checkpoints have no
    Engram owners) -- absence is reported, never fabricated."""
    if not isinstance(audit, dict) or "execution_actual" not in audit:
        return {"available": False, "reason": "no verified runtime audit recorded"}
    ranks = [r for r in (audit.get("ranks") or []) if isinstance(r, dict)]
    cache = audit.get("cache") or {}
    observed = cache.get("observed") or {}
    owners_in = audit.get("ngram_ram_owners") or {}
    mlock = audit.get("ngram_mlock") or {}
    locks_by = {(str(t.get("key")), t.get("device"), t.get("pid")): t
                for t in (mlock.get("tables") or [])}

    owners = {}
    for key, found in owners_in.items():
        if not found:
            continue
        o = found[0] if isinstance(found[0], dict) else {}
        lock = locks_by.get((str(key), o.get("device"), o.get("pid")))
        res = o.get("residency") if isinstance(o.get("residency"), dict) else None
        owners[key] = {
            "device": o.get("device"), "pid": o.get("pid"),
            "bytes": o.get("bytes"),
            "pages_resident": (res or {}).get("pages_resident"),
            "pages_total": (res or {}).get("pages_total"),
            "all_pages_resident": bool((res or {}).get("all_resident")),
            "mlocked": bool((lock or {}).get("is_locked")),
            "mlock_bytes": (lock or {}).get("locked_bytes"),
        }

    compact_observed = {}
    for label, geo in observed.items():
        if not isinstance(geo, dict):
            continue
        compact_observed[label] = {
            "layers_total": geo.get("layers_total"),
            "bytes_total": geo.get("bytes_total"),
            "cls_counts": geo.get("cls_counts"),
            "bits_variants": geo.get("bits_variants"),
        }

    return {
        "available": True,
        "ok": bool(audit.get("ok")),
        "path": audit.get("path"),
        "execution": {
            "actual": audit.get("execution_actual"),
            "loaded_tp": audit.get("loaded_tp"),
            "backend": {"requested": audit.get("requested_backend"),
                        "actual": audit.get("backend_actual"),
                        "rank_classes": {r.get("device"): r.get("backend_class") for r in ranks}},
        },
        "devices": {
            "expected": audit.get("expected_devices"),
            "output": audit.get("output_device"),
            "arch_expected": audit.get("arch_expected"),
            "parent_pid": audit.get("parent_pid"),
            "ranks": [{"device": r.get("device"), "pid": r.get("pid"),
                       "gcn_arch": r.get("gcnArchName"), "name": r.get("device_name"),
                       "torch_allocated_bytes": r.get("torch_allocated_bytes"),
                       "torch_peak_bytes": r.get("torch_peak_bytes"),
                       "torch_reserved_bytes": r.get("torch_reserved_bytes")} for r in ranks],
        },
        "cache_policy": {"requested": cache.get("requested"),
                          "observed": compact_observed,
                          "notes": cache.get("notes") or []},
        "mtp_placement": audit.get("mtp"),
        "ngram": {
            "expected_keys": audit.get("ngram_expected_keys"),
            "single_ram_owner": owners,
            "replicated_copies": audit.get("ngram_replicated_copies"),
            "mlock": {"env_requested": mlock.get("env_requested"),
                      "locked_total_bytes": mlock.get("locked_total_bytes"),
                      "records": mlock.get("records")},
        },
        "memory": audit.get("memory"),
        "cpu_helper": audit.get("cpu_helper"),
        "placement": audit.get("placement"),
        "problems": list(audit.get("problems") or []),
        "notes": list(audit.get("notes") or []),
    }


# ---------------------------------------------------------------------------
# Flag helpers for server.py (root adds these; kept here so dest names and
# validation stay in one place)
# ---------------------------------------------------------------------------

def add_helper_flags(parser):
    """The three runtime flags load_runtime()/PowerContext read:
      --power-socket    optional path of the privileged power helper
      --context-limit   cap n_ctx below the allocated cache (page-rounded)
      --max-output-tokens  server-side per-response output cap
    Existing -tp/-mtp/-cq/-ngr/-gs/-cs/-chunk_size/-mcl/-mcs flags keep coming
    from exllamav3.model_init.add_args unchanged."""
    parser.add_argument("--power-socket", dest="power_socket", type=str, default=None,
                        help="Unix socket of the privileged power helper; power policy is OFF when unset")
    parser.add_argument("--context-limit", dest="context_limit", type=int, default=None,
                        help="Advertise/serve at most this many context tokens (page-rounded, <= -cs)")
    parser.add_argument("--max-output-tokens", dest="max_output_tokens", type=int, default=None,
                        help="Server-side cap on generated tokens per response")
    return parser
