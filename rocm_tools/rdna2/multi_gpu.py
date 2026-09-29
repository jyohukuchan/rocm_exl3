#!/usr/bin/env python3
"""Layer-split (LS) helpers for the two-V620 (gfx1030) Phase-3 harness.

CPU-importable by design: nothing here imports torch or exllamav3 at module
scope. Functions that need torch take it as an argument, and the placement
auditor is duck-typed (any object with .modules / .device / .key / .caps /
.cache_layers), so the gate, planning and audit logic is fully unit-testable
on a CPU-only host (rocm_tools/rdna2/tests/test_ls_split_cpu.py).

What this is NOT:
  * not a monkeypatch or gate bypass -- the CLIs call these validators
    explicitly before touching a device, and the engine's own load path
    (Model.load with use_per_device) performs the actual splitting.
    Single-device paths keep their old gates: bench still requires exactly
    one visible GPU; the split gate requires the visible device count to
    EQUAL the number of budgets and every visible device to match
    --expect-arch.
  * not tensor parallelism -- this is the official layer-split autosplit API:
    model.load(use_per_device=[GiB, ...], max_chunk_size=...) with NO device
    argument (the engine asserts the two are mutually exclusive).

Units: use_per_device values are GiB (the engine converts int(gib * 1024**3))
and denote how much memory each visible GPU may USE for the split, on top of
whatever is already allocated at call time. All budgets must be finite and
strictly positive (a 0 would exclude the device, which is not the explicit
split that was requested). Explicit --use-per-device is the ONLY sanctioned
split path in these CLIs: without budgets the legacy single-device path runs
completely unchanged (bench requires exactly one visible GPU, while
collect_top1 loads onto its explicit --device). When a budgeted split loads, the actual placement
is AUDITED from the live objects -- module.device, each cache layer's device
against its owning attention module, the backing k/v tensors (plus the QSA
indexer's raw_k/pooled planes where the layer exposes them), the recurrent
layer states on hybrid modules (module.recurrent_layers: GDN conv_state/
recurrent_state, PLE conv_state/id_state), and model.active_devices. Budgets
and device counts are never evidence by themselves.

Phase 5 (Qwen3.8-Flash-Next / qwen4_exp hybrid architecture): the auditor
also records recurrent layer counts per device, and a KV-cache-ownership
problem is only raised for devices that actually own kv_cache-capable
modules -- a device holding only GDN/PLE layers must instead own its
recurrent states locally (they are audited against their owning module's
device; host-side holders such as PLE's id_state are recorded, not
enforced). Dense Qwen3 (all-attention) verdicts are unchanged. The n-gram
helpers (find_ngram_modules / describe_ngram_module / collect_ngram_state)
duck-type NGramEmbedding's load modes so bench.py can VERIFY -- not assume --
that a --ngram-ram request actually ended up with host-RAM-resident table
tensors; table residence is read from the tensors themselves, never inferred
from CUDA reserved memory (allocator reservation says nothing about host RAM
or board capacity).

All of it stays attribute-level introspection: no .item()/GPU sync, no
traversal of arbitrary Python object graphs (only .modules / .cache_layers /
.recurrent_layers / named tensor holders), so the audit is cheap at job
boundaries and unit-testable with stubs on a CPU-only host.
"""

from __future__ import annotations

import math
import re

# Safetensors-style transformer-layer key segment, e.g. "model.layers.17...".
# Embedding ("model.embed_tokens") and head ("lm_head" / logits_output) never
# match. layer_idx is NOT such a marker in this fork -- every module carries
# one -- so transformer identity is decided by capability flags plus this key.
_TRANSFORMER_KEY_RE = re.compile(r"(?:^|\.)layers\.\d+(?:\.|$|\b)")


# ---------------------------------------------------------------------------
# CLI-boundary budget validation (EXL3 layer-split only)
# ---------------------------------------------------------------------------

def validate_use_per_device(values) -> list[float] | None:
    """
    Validate --use-per-device budgets. None -> None (single-device path, or in
    a multi-device container the default autosplit; see plan_load_mode).
    Otherwise: at least two finite, strictly positive numbers, returned as a
    list of floats in GiB, in visible-device order (cuda:0, cuda:1, ...).
    Raises ValueError with a clear message (the CLIs wrap it into SystemExit).
    """
    if values is None:
        return None
    vals = list(values)
    if len(vals) < 2:
        raise ValueError(
            f"--use-per-device needs at least two budgets (one per visible GPU), got {len(vals)}")
    out: list[float] = []
    for i, v in enumerate(vals):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"--use-per-device[{i}] must be a number (GiB), got {v!r}")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"--use-per-device[{i}] must be finite, got {v!r}")
        if f <= 0.0:
            raise ValueError(f"--use-per-device[{i}] must be > 0 GiB (0 would exclude the "
                             f"device from the split), got {f}")
        out.append(f)
    return out


def reject_split_on_transformers(backend: str, budgets) -> None:
    """--use-per-device is an EXL3 layer-split option. The transformers
    backend loads the whole reference model onto one --device (cuda:0 or cpu);
    mixing the two is a user error, refused before any loading."""
    if budgets is not None and backend == "transformers":
        raise SystemExit(
            " !! FATAL: --use-per-device is EXL3 layer-split only; the transformers "
            "backend loads the whole model on one --device (use --backend exl3 for a split, "
            "or drop --use-per-device)")


# ---------------------------------------------------------------------------
# GPU identity capture + environment gate
# ---------------------------------------------------------------------------

def capture_device_identity(torch, index: int) -> dict:
    """
    Auditable identity of visible device `index`: logical index, name,
    gcnArchName, total memory, uuid and PCI bus id, all read directly from
    torch.cuda.get_device_properties(). No sysfs guessing: the container's
    torch exposes both uuid and pci_bus_id, and an unverified positional
    mapping could LIE about which card is which, so absent fields are recorded
    as null rather than inferred.
    """
    props = torch.cuda.get_device_properties(index)
    try:
        uuid = getattr(props, "uuid", None)
    except Exception:
        uuid = None
    try:
        pci = getattr(props, "pci_bus_id", None)
    except Exception:
        pci = None
    return {
        "index": index,
        "name": props.name,
        "gcnArchName": getattr(props, "gcnArchName", "") or "",
        "total_memory_bytes": getattr(props, "total_memory", None),
        "uuid": str(uuid) if uuid is not None else None,
        "pci_bus_id": str(pci) if pci is not None else None,
    }


def validate_split_gate(torch, expected_count: int, expect_arch: str) -> dict:
    """
    Layer-split environment gate (raises SystemExit, never proceeds):
      * torch must be a ROCm build (torch.version.hip set);
      * EXACTLY expected_count GPUs visible (== number of budgets, or == the
        visible count for the default split, which the caller proves is >= 2);
      * EVERY visible device's gcnArchName equals expect_arch -- one stray
        non-matching device must be hidden, not tolerated.
    Returns per-device identities for the artifact.
    """
    if getattr(torch.version, "hip", None) is None:
        raise SystemExit(" !! FATAL: torch build has no ROCm (torch.version.hip is None); "
                         "run inside the rocm-exl3-rdna2 container venv")
    n = torch.cuda.device_count()
    if n != expected_count:
        raise SystemExit(
            f" !! FATAL: layer split with {expected_count} expected GPU(s) found {n} visible. "
            f"Budgets map 1:1 to visible order, so check CUDA_VISIBLE_DEVICES/"
            f"HIP_VISIBLE_DEVICES -- never a partial split, never extra devices")
    if not torch.cuda.is_available():
        raise SystemExit(f" !! FATAL: torch.cuda.is_available() is False despite "
                         f"{n} visible devices")
    devices = []
    for i in range(n):
        ident = capture_device_identity(torch, i)
        if ident["gcnArchName"] != expect_arch:
            raise SystemExit(
                f" !! FATAL: device {i} ({ident['name']}) has gcnArchName "
                f"{ident['gcnArchName']!r}, expected {expect_arch!r}: the layer-split gate "
                f"requires EVERY visible GPU to be the {expect_arch} V620 "
                f"(a stray non-matching device must be hidden, not tolerated)")
        devices.append(ident)
    return {
        "mode": "layer_split",
        "visible_devices": n,
        "expect_arch": expect_arch,
        "torch_version_hip": torch.version.hip,
        "devices": devices,
    }


def plan_load_mode(torch, budgets, expect_arch: str) -> dict:
    """
    Decide HOW to load from --use-per-device only (explicit budgets are the
    ONE sanctioned split path in these CLIs; the engine's default autosplit
    is deliberately not exposed as a no-flag variant):

      budgets given -> ("layer_split", budgets, gate requiring exactly
                        len(budgets) visible gfx1030 devices)
      no budgets    -> ("single", None, None)  legacy path fully unchanged:
                        bench's own gate still demands exactly one visible
                        GPU; collect_top1 keeps its explicit --device load

    On the layer_split branch model.load() MUST be called without a device
    argument: with budgets -> use_per_device=[...]; the caller then audits
    the actual placement against [0 .. visible-1].
    """
    if budgets is not None:
        gate = validate_split_gate(torch, len(budgets), expect_arch)
        gate["use_per_device_gib"] = list(budgets)
        return {"mode": "layer_split", "budgets": list(budgets), "gate": gate}
    return {"mode": "single", "budgets": None, "gate": None}


# ---------------------------------------------------------------------------
# per-device memory + transfer stats
# ---------------------------------------------------------------------------

def device_key(i: int) -> str:
    return f"cuda:{i}"


def sync_devices(torch, indices: list[int]) -> None:
    """Synchronize every device the split uses. Job boundaries only."""
    for i in indices:
        torch.cuda.synchronize(torch.device(device_key(i)))


def reset_peak_on_devices(torch, indices: list[int]) -> None:
    """Reset peak-memory stats on every device the split uses (job boundary only)."""
    for i in indices:
        torch.cuda.reset_peak_memory_stats(torch.device(device_key(i)))


def device_memory_snapshot(torch, indices: list[int]) -> dict:
    """
    Per-device peak/allocated/reserved bytes, keyed "cuda:i" for stable JSON.
    Reads current stats (no sync/reset here -- the caller owns boundaries).
    """
    snap: dict = {}
    for i in indices:
        dev = torch.device(device_key(i))
        snap[device_key(i)] = {
            "peak_bytes": torch.cuda.max_memory_allocated(dev),
            "allocated_bytes": torch.cuda.memory_allocated(dev),
            "reserved_bytes": torch.cuda.memory_reserved(dev),
        }
    return snap


def diff_transfer_stats(before: dict, after: dict) -> dict:
    """exllamav3.util.device_copy.stats is cumulative {direct, bounced, probes};
    return the per-job delta (only keys seen on either side)."""
    keys = sorted(set(before) | set(after))
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in keys}


# ---------------------------------------------------------------------------
# n-gram (PLE) table residence: RAM vs disk-streamed, measured from tensors
# ---------------------------------------------------------------------------

# NGramEmbedding.load() sets exactly one of these four modes; RAM modes keep
# the table as CPU tensors in .tables, disk modes keep DiskTensorHandles in
# .handles and never load the table. No other module uses this vocabulary.
NGRAM_MODES = ("trellis_disk", "trellis_ram", "fp16_disk", "fp16_ram")
NGRAM_RAM_MODES = ("trellis_ram", "fp16_ram")


def _tensor_bytes(t) -> int | None:
    """numel * element_size from tensor METADATA only (never .item(), never a
    device query that syncs); None when the object doesn't speak tensor."""
    try:
        return int(t.numel()) * int(t.element_size())
    except Exception:
        return None


def find_ngram_modules(model) -> list:
    """Every loaded NGramEmbedding in the live model, duck-typed by its
    load-mode string (multi_gpu stays exllamav3-free). Walks the .modules
    tree only."""
    out = []
    for top in (getattr(model, "modules", None) or []):
        for m in _iter_module_tree(top):
            if (getattr(m, "mode", None) in NGRAM_MODES
                    or type(m).__name__ == "NGramEmbedding"):
                out.append(m)
    return out


def describe_ngram_module(m) -> dict:
    """
    Auditable residence record for one NGramEmbedding: the ACTUAL mode after
    load, where the backing tensors really live (str(tensor.device), metadata
    byte counts), how many disk handles exist if any, and a copy of the
    cumulative prefetch_stats counters when the module exposes them. This is
    the evidence a --ngram-ram assumption is checked against; CUDA
    reserved/allocated memory is NOT evidence of table residence.
    """
    tables = list(getattr(m, "tables", None) or [])
    handles = list(getattr(m, "handles", None) or [])
    tinfo = [{"device": str(getattr(t, "device", None)), "bytes": _tensor_bytes(t)}
             for t in tables]
    known = [x["bytes"] for x in tinfo if x["bytes"] is not None]
    ram_bytes = sum(known) if tables and len(known) == len(tables) else None
    stats = getattr(m, "prefetch_stats", None)
    mode = getattr(m, "mode", None)
    return {
        "key": getattr(m, "key", None),
        "mode": mode,
        "device": str(getattr(m, "device", None)),
        "ram_backed": bool(tables) and mode in NGRAM_RAM_MODES,
        "num_rows": getattr(m, "num_rows", None),
        "rows_per_shard": getattr(m, "rows_per_shard", None),
        "num_table_tensors": len(tables),
        "table_tensors": tinfo,
        "table_residence": sorted({x["device"] for x in tinfo}),
        "table_ram_bytes": ram_bytes,
        "num_disk_handles": len(handles),
        "disk_handle_files": sorted({str(getattr(h, "filename", "?")) for h in handles}),
        "prefetch_stats": dict(stats) if isinstance(stats, dict) else None,
    }


def collect_ngram_state(model, require_ram: bool) -> dict:
    """
    Snapshot + verdict for every n-gram table in the live model. With
    require_ram (bench --ngram-ram) a table-bearing model that did NOT end up
    RAM-backed (disk mode, missing table tensors, or a RAM-mode table whose
    tensors are not host-resident) is reported in "problems" -- the caller
    fails the run closed. Without the requirement the record is observational
    only (legacy default unchanged: disk modes pass). A model with no tables
    at all never fails; the no-op is surfaced in "notes".
    """
    modules = [describe_ngram_module(m) for m in find_ngram_modules(model)]
    problems: list[str] = []
    notes: list[str] = []
    if require_ram and not modules:
        notes.append("--ngram-ram requested but this model exposes no n-gram table "
                     "(flag is a no-op; nothing was forced into RAM)")
    for rec in modules:
        if not require_ram:
            continue
        if rec["mode"] not in NGRAM_RAM_MODES:
            problems.append(
                f"{rec['key']}: n-gram table is {rec['mode']!r} (disk-streamed), not "
                f"RAM-backed, although --ngram-ram set ngram_stream_from_disk=False")
        elif not rec["num_table_tensors"]:
            problems.append(f"{rec['key']}: RAM mode {rec['mode']!r} but no table "
                            f"tensors are present")
        else:
            off_host = sorted({d for d in rec["table_residence"] if d != "cpu"})
            if off_host:
                problems.append(f"{rec['key']}: RAM-mode table tensors are not host-"
                                f"resident: {off_host}")
    per_bytes = [r["table_ram_bytes"] for r in modules]
    total_ram = (sum(b for b in per_bytes if b is not None)
                 if modules and all(b is not None for b in per_bytes) else None)
    return {
        "requested": bool(require_ram),
        "tables_found": len(modules),
        "ram_backed": (all(r["mode"] in NGRAM_RAM_MODES for r in modules)
                       if modules else None),
        "total_table_ram_bytes": total_ram,
        "modules": modules,
        "problems": problems,
        "notes": notes,
        "ok": not problems,
        "residence_evidence": "read from the actual table tensors (device + metadata "
                              "bytes) and process RSS/fault snapshots in the artifact; "
                              "CUDA 'reserved' is allocator state on the board and is "
                              "NOT treated as evidence of table residence or capacity",
    }


# ---------------------------------------------------------------------------
# placement audit: does the split ACTUALLY live where it was supposed to?
# ---------------------------------------------------------------------------

def _iter_module_tree(module):
    """Recursively yield a module and all its children (Module.__iter__ shape)."""
    yield module
    for sub in getattr(module, "modules", None) or []:
        yield from _iter_module_tree(sub)


def _caps(module) -> dict:
    return getattr(module, "caps", None) or {}


def _has_cache_caps(module) -> bool:
    return bool(_caps(module).get("kv_cache") or _caps(module).get("recurrent_cache"))


def _is_transformer(module) -> bool:
    """
    A top-level module is a REAL transformer layer only if its tree carries
    attention/recurrent cache capability or a transformer-layer key. Never
    layer_idx: in this fork embed and head carry one too, so trusting it (or
    counting distinct devices) would let an embed/head-only placement pass as
    a split.
    """
    for m in _iter_module_tree(module):
        if _caps(m).get("logits_output"):
            continue
        if _has_cache_caps(m):
            return True
        key = getattr(m, "key", None)
        if key and _TRANSFORMER_KEY_RE.search(str(key)):
            return True
    return False


def _cache_layer_tensors(cl) -> dict:
    """The actual backing tensors of one cache layer, read from the object
    itself (CacheLayer_fp16 exposes .k/.v directly; the QSA layers expose the
    indexer planes as .raw_k/.pooled; fall back to get_tensors(), which for
    CacheLayer_quant lists the packed qk/qv/sk/sv -- and the planes through
    the QSAPlanes mixin override)."""
    tensors: dict = {}
    for attr in ("k", "v", "raw_k", "pooled"):
        t = getattr(cl, attr, None)
        if t is not None:
            tensors[attr] = t
    if not tensors:
        try:
            got = cl.get_tensors()
        except Exception:
            got = None
        if got:
            for idx, t in enumerate(got if isinstance(got, (list, tuple)) else ()):
                if t is not None:
                    tensors[f"t{idx}"] = t
    return tensors


def _recurrent_layer_tensors(rl) -> dict:
    """Named backing tensors of one recurrent layer state (GDNLayerState:
    conv_state/recurrent_state; PLELayerState: conv_state/id_state), with a
    get_state_tensors() fallback for state classes using other names. Direct
    attributes only -- no arbitrary object-graph traversal."""
    tensors: dict = {}
    for attr in ("conv_state", "recurrent_state", "id_state"):
        t = getattr(rl, attr, None)
        if t is not None:
            tensors[attr] = t
    if not tensors:
        try:
            got = rl.get_state_tensors()
        except Exception:
            got = None
        if got:
            for idx, t in enumerate(got if isinstance(got, (list, tuple)) else ()):
                if t is not None:
                    tensors[f"t{idx}"] = t
    return tensors


def audit_placement(model, expect_indices: list[int]) -> dict:
    """
    Verify the ACTUAL placement after a layer-split load -- read from the live
    objects, never inferred from budgets or device counts:

      * every module must have a non-None .device inside the expected set
        (CPU is allowed only for prefer_cpu modules); submodule devices must
        agree with their top-level module device; unexpected device indices
        are recorded as problems and EXCLUDED from the progression sequence
        (this function never raises);
      * transformer ownership (capability/key based, see _is_transformer,
        audited by MODULE count, not device count): when the placement holds
        at least one transformer module, every expected device must own at
        least one -- embed/head-only cards, or a split that never advanced,
        are rejected; a degenerate model with zero transformer modules is
        audited by module count and passes vacuously;
      * when cache layers exist, each cache layer must have backing storage
        (k/v tensors, QSA indexer planes raw_k/pooled, or non-empty
        get_tensors()), its recorded .device must equal its OWNING attention
        module's .device, its actual tensor devices must match that device,
        and the device must be in the split; every expected device that owns
        kv_cache-capable modules must own at least one cache layer (KV pages
        local to the attention that needs them). A device that owns NO
        kv_cache-capable modules (e.g. only GDN/PLE hybrid layers) is not
        held to the ordinary-attention count;
      * recurrent layer states (module.recurrent_layers: GDN conv_state /
        recurrent_state, PLE conv_state / id_state) are audited the same way:
        allocated (.device not None), local to their owning module's device,
        inside the split; computation tensors must be on that device, while
        PLE's id_state must stay on CPU for the host-side n-gram hashing.
        Empty or meta-only state is rejected. Every expected device that owns recurrent_cache-capable
        modules must own at least one recurrent state, and the counts per
        device are recorded;
      * device progression along the forward order (model.modules) must be
        contiguous: one run per device in expected order -- (0,1,0) means
        placement is broken;
      * model.output_device and model.active_devices are cross-checked.

    Returns an auditable dict (ordered per-module, per-cache-layer and
    per-recurrent-layer records) with "ok" and "problems". Never raises: the
    CLIs turn problems into a hard, fail-closed exit. All reads are attribute
    / tensor-metadata level (numel, element_size, .device): no .item() or
    other GPU sync, no traversal beyond .modules / .cache_layers /
    .recurrent_layers.
    """
    expected = list(expect_indices)
    order_of = {i: k for k, i in enumerate(expected)}
    problems: list[str] = []
    modules_out: list[dict] = []
    cache_layers_out: list[dict] = []
    recurrent_layers_out: list[dict] = []
    dev_of_top: list[int | None] = []
    transformer_own: dict[str, int] = {device_key(i): 0 for i in expected}
    cache_own: dict[str, int] = {device_key(i): 0 for i in expected}
    recurrent_own: dict[str, int] = {device_key(i): 0 for i in expected}
    kv_modules_own: dict[str, int] = {device_key(i): 0 for i in expected}
    rec_modules_own: dict[str, int] = {device_key(i): 0 for i in expected}
    total_cache_layers = 0
    total_recurrent_layers = 0

    modules = list(getattr(model, "modules", None) or [])
    if not modules:
        problems.append("model.modules is empty after load -- nothing to audit")

    for top_i, module in enumerate(modules):
        dev = getattr(module, "device", None)
        tree = list(_iter_module_tree(module))
        top_dev_str = str(dev) if dev is not None else None
        mismatched: list[str] = []
        sub_devices: dict[str, int] = {}
        cache_keys: list[str] = []
        for sm in tree:
            if sm is module:
                continue
            sd = getattr(sm, "device", None)
            skey = str(sd) if sd is not None else "None"
            sub_devices[skey] = sub_devices.get(skey, 0) + 1
            if sd is not None and dev is not None and sd != dev:
                mismatched.append(f"{getattr(sm, 'key', '?')}@{sd}")
            if _has_cache_caps(sm):
                cache_keys.append(str(getattr(sm, "key", "?")))

        idx = dev.index if (dev is not None and getattr(dev, "type", None) == "cuda") else None
        caps = _caps(module)
        transformer = _is_transformer(module)
        inside = idx is not None and idx in order_of
        # capability tallies (full tree, the top module included: PLELayer is
        # itself recurrent_cache-capable and lives at the top level)
        n_kv_caps = sum(1 for sm in tree if _caps(sm).get("kv_cache"))
        n_rec_caps = sum(1 for sm in tree if _caps(sm).get("recurrent_cache"))
        if inside:
            kv_modules_own[device_key(idx)] += n_kv_caps
            rec_modules_own[device_key(idx)] += n_rec_caps

        if dev is None:
            problems.append(f"module[{top_i}] {getattr(module, 'key', '?')}: device is None "
                            f"(module did not end up loaded on any device)")
        elif getattr(dev, "type", "cpu") == "cpu":
            if not caps.get("prefer_cpu"):
                problems.append(f"module[{top_i}] {getattr(module, 'key', '?')}: on CPU without "
                                f"a prefer_cpu capability (unexpected for the split)")
        elif not inside:
            problems.append(f"module[{top_i}] {getattr(module, 'key', '?')}: on {top_dev_str}, "
                            f"not one of the expected split devices {expected}")
        if mismatched:
            problems.append(f"module[{top_i}] {getattr(module, 'key', '?')}: submodule device(s) "
                            f"disagree with the module device: {mismatched[:8]}")

        if transformer and inside:
            transformer_own[device_key(idx)] += 1
        if inside:
            dev_of_top.append(idx)
        # unexpected/None/cpu devices stay out of the progression sequence --
        # they are already problems; order_of[idx] must never be indexed with
        # a device that is not part of the split (no KeyError: audit reports).

        modules_out.append({
            "order": top_i,
            "key": getattr(module, "key", None),
            "layer_idx": getattr(module, "layer_idx", None),
            "device": top_dev_str,
            "device_index": idx if inside else None,
            "outside_expected": bool(idx is not None and not inside),
            "num_modules": len(tree),
            "submodule_devices": sub_devices,
            "transformer": transformer,
            "cache_modules": cache_keys,
            "kv_cache_modules": n_kv_caps,
            "recurrent_cache_modules": n_rec_caps,
        })

        for m in tree:
            for cl in (getattr(m, "cache_layers", None) or []):
                total_cache_layers += 1
                cd = getattr(cl, "device", None)
                cidx = cd.index if (cd is not None and getattr(cd, "type", None) == "cuda") else None
                mdev = getattr(m, "device", None)
                tensors = _cache_layer_tensors(cl)
                tensor_devices = {name: str(getattr(t, "device", None))
                                  for name, t in tensors.items()}
                ckey = device_key(cidx) if cidx in order_of else None

                problems_before = len(problems)
                if cd is None:
                    problems.append(f"cache layer of {getattr(m, 'key', '?')}: device is None "
                                    f"(cache tensors were not allocated on any device)")
                elif cidx is not None and cidx not in order_of:
                    problems.append(f"cache layer of {getattr(m, 'key', '?')}: on {cd}, not one "
                                    f"of the expected split devices {expected}")
                if mdev is not None and cd is not None and cd != mdev:
                    problems.append(f"cache layer of {getattr(m, 'key', '?')}: records device "
                                    f"{cd} but its owning attention module is on {mdev} -- "
                                    f"KV pages would be remote to their layer")
                if not tensors:
                    problems.append(f"cache layer of {getattr(m, 'key', '?')}: no backing k/v "
                                    f"tensors (empty cache storage on a cache-capable module)")
                elif hasattr(cl, "k") and hasattr(cl, "v") and (cl.k is None or cl.v is None):
                    problems.append(f"cache layer of {getattr(m, 'key', '?')}: missing k/v tensor")
                elif any(hasattr(cl, name) and getattr(cl, name) is None
                         for name in ("raw_k", "pooled")):
                    problems.append(f"cache layer of {getattr(m, 'key', '?')}: missing QSA plane")
                elif cd is not None:
                    for name, td in sorted(tensor_devices.items()):
                        if td != str(cd):
                            problems.append(f"cache layer of {getattr(m, 'key', '?')}: records "
                                            f"device {cd} but tensor {name} is on {td}")
                            break
                if len(problems) == problems_before and ckey is not None:
                    cache_own[ckey] += 1
                tbytes = [_tensor_bytes(t) for t in tensors.values()]
                cache_layers_out.append({
                    "module_key": getattr(m, "key", None),
                    "module_device": str(mdev) if mdev is not None else None,
                    "layer_idx": getattr(m, "layer_idx", None),
                    "device": str(cd) if cd is not None else None,
                    "device_index": cidx if cidx in order_of else None,
                    "n_tensors": len(tensors),
                    "tensor_devices": tensor_devices,
                    "total_bytes": (sum(b for b in tbytes if b is not None)
                                    if tensors and all(b is not None for b in tbytes) else None),
                })

            for rl in (getattr(m, "recurrent_layers", None) or []):
                total_recurrent_layers += 1
                rd = getattr(rl, "device", None)
                ridx = rd.index if (rd is not None and getattr(rd, "type", None) == "cuda") else None
                mdev = getattr(m, "device", None)
                tensors = _recurrent_layer_tensors(rl)
                tensor_devices = {name: str(getattr(t, "device", None))
                                  for name, t in tensors.items()}
                rkey = device_key(ridx) if ridx in order_of else None
                rlabel = f"recurrent layer state of {getattr(m, 'key', '?')}"

                problems_before = len(problems)
                if rd is None:
                    problems.append(f"{rlabel}: device is None (state tensors were not "
                                    f"allocated on any device)")
                elif ridx is not None and ridx not in order_of:
                    problems.append(f"{rlabel}: on {rd}, not one of the expected split "
                                    f"devices {expected}")
                elif ridx is None and getattr(rd, "type", None) not in (None, "cuda"):
                    problems.append(f"{rlabel}: on {rd}, not a cuda device of the split "
                                    f"{expected}")
                if mdev is not None and rd is not None and rd != mdev:
                    problems.append(f"{rlabel}: records device {rd} but its owning module is "
                                    f"on {mdev} -- state would be remote to the layer that "
                                    f"advances it")
                # PLE's id history is intentionally on the CPU. Computation
                # state must actually be allocated on its owning GPU.
                if not tensors:
                    problems.append(f"{rlabel}: no backing state tensors")
                if rd is not None:
                    remote = sorted({td for name, td in tensor_devices.items()
                                     if td != ("cpu" if name == "id_state" else str(rd))})
                    if remote:
                        problems.append(f"{rlabel}: records device {rd} but state tensor(s) "
                                        f"live on {remote}")
                if len(problems) == problems_before and rkey is not None:
                    recurrent_own[rkey] += 1
                rbytes = [_tensor_bytes(t) for t in tensors.values()]
                recurrent_layers_out.append({
                    "module_key": getattr(m, "key", None),
                    "module_device": str(mdev) if mdev is not None else None,
                    "layer_idx": getattr(m, "layer_idx", None),
                    "state_class": type(rl).__name__,
                    "device": str(rd) if rd is not None else None,
                    "device_index": ridx if ridx in order_of else None,
                    "n_tensors": len(tensors),
                    "tensor_devices": tensor_devices,
                    "total_bytes": (sum(b for b in rbytes if b is not None)
                                    if tensors and all(b is not None for b in rbytes) else None),
                })

    total_transformers = sum(transformer_own.values())
    if total_transformers > 0:
        for i in expected:
            if transformer_own[device_key(i)] == 0:
                problems.append(f"device {device_key(i)} owns NO transformer modules (0 of "
                                f"{total_transformers} split across the other devices) -- the "
                                f"model did not actually split across the requested devices "
                                f"(budgets are not evidence; placement is)")
    if total_cache_layers > 0:
        # ordinary-attention count: only devices that actually own kv_cache-
        # capable modules must own KV pages locally. A hybrid split may put
        # ONLY GDN/PLE layers on a device; holding it to the attention count
        # would reject a placement that is correct.
        for i in expected:
            k = device_key(i)
            if kv_modules_own[k] > 0 and cache_own[k] == 0:
                problems.append(f"device {k} owns NO cache layers although "
                                f"{total_cache_layers} cache layer(s) exist -- KV pages are "
                                f"not local to the devices whose attention layers need them")
    if total_recurrent_layers > 0:
        # the hybrid counterpart: every device whose layers ADVANCE recurrent
        # state (GDN conv+delta state, PLE conv+id state) must own it locally
        for i in expected:
            k = device_key(i)
            if rec_modules_own[k] > 0 and recurrent_own[k] == 0:
                problems.append(f"device {k} owns NO recurrent layer states although "
                                f"{total_recurrent_layers} recurrent state(s) exist -- "
                                f"hybrid (GDN/PLE) state is not local to the devices whose "
                                f"layers advance it")

    # contiguous device progression over the forward order (only in-split,
    # cuda-mounted top modules join the sequence)
    seq = dev_of_top
    ranks = [order_of[idx] for idx in seq]
    bad = [(seq[k], seq[k + 1]) for k in range(len(ranks) - 1) if ranks[k + 1] < ranks[k]]
    if bad:
        problems.append(f"device progression is not contiguous (expected one run per device in "
                        f"budget order): regressions {bad[:8]} in sequence {seq}")

    active = [i if not hasattr(i, "index") else i.index
              for i in (getattr(model, "active_devices", None) or [])]
    if sorted(active) != sorted(expected):
        problems.append(f"model.active_devices is {active}, expected exactly the split "
                        f"devices {expected} -- the loader did not use all requested devices")

    out_dev = getattr(model, "output_device", None)
    if out_dev is not None and getattr(out_dev, "type", None) == "cuda" \
            and out_dev.index not in order_of:
        problems.append(f"model.output_device is {out_dev}, outside the expected split "
                        f"devices {expected}")

    runs: list[dict] = []
    for idx in seq:
        if runs and runs[-1]["device"] == device_key(idx):
            runs[-1]["modules"] += 1
        else:
            runs.append({"device": device_key(idx), "modules": 1})

    return {
        "ok": not problems,
        "expected_devices": [device_key(i) for i in expected],
        "active_devices": active,
        "device_progression": runs,
        "transformer_modules_per_device": transformer_own,
        "cache_layers_per_device": cache_own,
        "total_cache_layers": total_cache_layers,
        "recurrent_layers_per_device": recurrent_own,
        "total_recurrent_layers": total_recurrent_layers,
        "kv_cache_modules_per_device": kv_modules_own,
        "recurrent_cache_modules_per_device": rec_modules_own,
        "total_transformer_modules": total_transformers,
        "output_device": str(out_dev) if out_dev is not None else None,
        "modules": modules_out,
        "cache_layers": cache_layers_out,
        "recurrent_layers": recurrent_layers_out,
        "problems": problems,
    }
