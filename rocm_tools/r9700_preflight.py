#!/usr/bin/env python3
"""r9700_preflight.py -- read-only environment preflight for the R9700 (gfx1201).

Diagnoses the environment *before* building or loading EXL3 on an AMD Radeon
AI PRO R9700 (gfx1201). This is an environment diagnostic, not a kernel port:
it proves nothing about arbitrary RDNA4 inference working. It

  * lazily imports torch ONLY for a hardware query (device properties),
  * never imports exllamav3, never imports/JITs the native extension,
    never loads a model, launches GPU kernels, runs subprocesses,
    installs anything or mutates the environment,
  * reports torch/HIP versions, the selected GPU's name, normalized arch
    (feature suffix stripped from gcnArchName), VRAM bytes and
    shared_memory_per_block,
  * checks the gfx1201 64 KiB LDS expectation (measured 65536 in this repo,
    see setup.py GPU_ARCH_SMEM and doc/r9700_vs_v620.md),
  * records only relevant build / fused-MoE environment settings (allowlist),
  * warns (blocks) on fused-MoE forcing via EXL3_ROCM_RDNA4_FUSED_MOE on
    gfx120x, matching exllamav3/rocm_py/__init__.py's _env_on semantics,
  * compares the *build-target environment* (PYTORCH_ROCM_ARCH / GPU_ARCHS)
    against the device, while stating explicitly that environment is NOT
    evidence of the architecture of an already compiled binary. Binary
    target verification therefore stays "unknown".

A valid R9700 fact set is labelled status "comparison_workarounds_required"
(not ready, not general_supported): the measured R9700 results in
doc/r9700_vs_v620.md required external comparison adapters that are not part
of this library. Exit code 0 means "the report was collected", not "inference
is proven".

--snapshot FILE replays an offline JSON fact file (schema:
doc/r9700_preflight.md) so the same analysis runs CPU-only and
reproducibly. The tool itself only reads inputs; its report goes to stdout.

Usage:
    python3 rocm_tools/r9700_preflight.py                 # live query
    python3 rocm_tools/r9700_preflight.py --device 1      # other GPU index
    python3 rocm_tools/r9700_preflight.py --json          # machine-readable
    python3 rocm_tools/r9700_preflight.py --snapshot snap.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

TOOL_NAME = "r9700_preflight"
REPORT_VERSION = 1
SNAPSHOT_KIND = "r9700_preflight_snapshot"
SNAPSHOT_SCHEMA_VERSION = 1

# Exit codes (finite, documented).
EXIT_OK = 0            # report collected (does NOT prove inference)
EXIT_USAGE = 1         # bad snapshot file / invalid device selection
EXIT_NO_HIP = 2        # torch missing, non-ROCm torch, or no HIP device visible
EXIT_BLOCKED_ENV = 3   # unsafe fused-MoE forcing on gfx120x

# Mirrors of the build-time facts this tool diagnoses. Kept as local data so
# the tool runs without importing setup.py (which shells out at import time).
# If setup.py's tables change, update these (and the tests).
SUPPORTED_GPU_ARCHS = {
    # setup.py SUPPORTED_GPU_ARCHS
    "gfx1030",
    "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201",
}
GPU_ARCH_SMEM = {
    # setup.py GPU_ARCH_SMEM: LDS bytes per workgroup the EXL3 kernels may assume.
    "gfx1030": 65536,
    "gfx1100": 92160, "gfx1101": 92160, "gfx1102": 92160,
    "gfx1150": 65536, "gfx1151": 65536,
    "gfx1200": 92160,
    "gfx1201": 65536,  # R9700: measured 64 KB per workgroup (doc/r9700_vs_v620.md)
}
DEFAULT_ARCH_SMEM = 65536  # unknown arch: conservative, as in setup.py
MIN_ROCM = (7, 2, 4)       # setup.py MIN_ROCM

# Allowlist: only these environment variables are ever recorded (build target
# selection + fused-MoE steering). Nothing else, no secrets, no full environ.
RELEVANT_ENV_KEYS = (
    "PYTORCH_ROCM_ARCH",            # setup.py _resolve_offload_archs
    "GPU_ARCHS",                    # setup.py _resolve_offload_archs fallback
    "ROCM_PATH",                    # setup.py _check_rocm_version
    "EXL3_SKIP_ROCM_VERSION_CHECK", # setup.py ROCm version gate override
    "EXL3_ROCM_RDNA4_FUSED_MOE",    # rocm_py gfx120x fused-MoE steer bypass
    "EXL3_ROCM_MOE_MGEMM_ROUTE",    # rocm_py MoE decode route
    "EXL3_ROCM_MOE_BSZN",           # rocm_py MoE bszN route
    "EXL3_ROCM_MOE_MGEMM_MAX_ROWS", # rocm_py MoE row cap
    "EXL3_ROCM_MOE_DISABLE",        # rocm_py MoE dense fallback
)


class PreflightError(Exception):
    """Fatal, user-facing problem with a given exit code and diagnostics."""

    def __init__(self, exit_code: int, status: str, messages: list[str]):
        super().__init__("; ".join(messages))
        self.exit_code = exit_code
        self.status = status
        self.messages = messages


def env_on(env: dict, name: str, default: bool = False) -> bool:
    """Copy of exllamav3/rocm_py/__init__.py::_env_on semantics.

    Unset -> default; otherwise the stripped value is ON unless it is one of
    "", "0", "false", "False". (Note: "no" and "FALSE" count as ON -- matching
    the runtime exactly is the point of this diagnostic.)
    """
    v = env.get(name)
    if v is None:
        return default
    return v.strip() not in ("", "0", "false", "False")


def normalize_arch(gcn_arch_name: str) -> tuple[str, str]:
    """Split torch's gcnArchName into (base arch, feature suffix).

    "gfx1201:sramecc+:xnack-" -> ("gfx1201", "sramecc+:xnack-")
    """
    raw = gcn_arch_name.strip()
    arch, _, features = raw.partition(":")
    return arch, features


def resolve_build_targets(env: dict) -> tuple[list[str], str | None]:
    """Parse the build-target env exactly like setup.py::_resolve_offload_archs.

    PYTORCH_ROCM_ARCH wins over GPU_ARCHS; empty string is falsy and falls
    through. Tokens are comma- or whitespace-separated; each is normalized by
    stripping a ":feature" suffix. Returns (archs, source_var); ([], None)
    means no explicit target -> hipcc/rocminfo auto-detect at build time.
    """
    raw = env.get("PYTORCH_ROCM_ARCH") or env.get("GPU_ARCHS")
    if not raw:
        return [], None
    source = "PYTORCH_ROCM_ARCH" if env.get("PYTORCH_ROCM_ARCH") else "GPU_ARCHS"
    tokens = [t for t in raw.replace(",", " ").split() if t]
    return [normalize_arch(t)[0] for t in tokens], source


def read_rocm_version(env: dict) -> tuple[list[int] | None, str | None, str]:
    """Read $ROCM_PATH/.info/version like setup.py::_rocm_version. File read only.

    An unreadable, missing, or non-UTF8 version file is an unavailable fact
    (parsed/version come back None), never a raised exception: setup.py
    degrades to a warning in the same situation.
    """
    rocm_path = env.get("ROCM_PATH") or "/opt/rocm"
    path = os.path.join(rocm_path, ".info", "version")
    try:
        with open(path, encoding="utf8") as fp:
            raw = fp.read().strip()
    except (OSError, UnicodeDecodeError):
        return None, None, path
    head = raw.split("-")[0].strip()
    parts = head.split(".")
    try:
        return [int(p) for p in parts[:3]], raw, path
    except ValueError:
        return None, raw, path


def collect_live_facts(env: dict) -> dict:
    """Query hardware facts. torch is imported lazily and ONLY here, and only
    to read device properties. Nothing else in this tool touches GPU state."""
    facts: dict = {
        "kind": SNAPSHOT_KIND,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "collected_via": "live torch query",
        "torch": {"importable": False, "version": None, "hip_version": None,
                  "cuda_is_available": False, "device_count": 0},
        "rocm": None,
        "devices": [],
        "env": {k: env[k] for k in RELEVANT_ENV_KEYS if k in env},
    }
    try:
        import torch  # lazy, hardware query only
    except Exception as e:  # ImportError and anything else that hides torch
        facts["torch"]["import_error"] = f"{type(e).__name__}: {e}"
    else:
        hip = getattr(getattr(torch, "version", None), "hip", None)
        cuda = getattr(torch, "cuda", None)
        try:
            available = bool(cuda is not None and cuda.is_available())
            count = int(cuda.device_count()) if available else 0
        except Exception as e:
            facts["torch"] = {"importable": True,
                              "version": getattr(torch, "__version__", None),
                              "hip_version": hip,
                              "cuda_is_available": False, "device_count": 0,
                              "query_error": f"{type(e).__name__}: {e}"}
        else:
            facts["torch"] = {"importable": True,
                              "version": getattr(torch, "__version__", None),
                              "hip_version": hip,
                              "cuda_is_available": available,
                              "device_count": count}
            for i in range(count):
                try:
                    p = cuda.get_device_properties(i)
                    facts["devices"].append({
                        "index": i,
                        "name": getattr(p, "name", None),
                        "gcn_arch_name": getattr(p, "gcnArchName", None),
                        "total_memory": getattr(p, "total_memory", None),
                        "shared_memory_per_block":
                            getattr(p, "shared_memory_per_block", None),
                    })
                except Exception as e:
                    facts["devices"].append({
                        "index": i, "name": None, "gcn_arch_name": None,
                        "total_memory": None, "shared_memory_per_block": None,
                        "query_error": f"{type(e).__name__}: {e}",
                    })
    ver, raw, path = read_rocm_version(env)
    facts["rocm"] = {"version": raw, "parsed": ver, "version_file": path}
    return facts


# ---------------------------------------------------------------------------
# Snapshot loading and validation
# ---------------------------------------------------------------------------

def _need(obj: dict, key: str, where: str, problems: list[str]):
    if key not in obj:
        problems.append(f"{where}: missing required field '{key}'")
        return None
    return obj[key]


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def validate_facts(facts: object, source: str) -> dict:
    """Schema-check a facts dict (from snapshot or live collection).

    Raises PreflightError(EXIT_USAGE, "invalid_snapshot", [...]) listing every
    problem found, with dot-paths, so one run gives the whole fix list.
    Returns the validated dict.
    """
    problems: list[str] = []
    if not isinstance(facts, dict):
        raise PreflightError(EXIT_USAGE, "invalid_snapshot",
                             [f"{source}: snapshot must be a JSON object, "
                              f"got {type(facts).__name__}"])
    if facts.get("kind") != SNAPSHOT_KIND:
        problems.append(f"{source}: 'kind' must be {SNAPSHOT_KIND!r}, "
                        f"got {facts.get('kind')!r}")
    sv = facts.get("schema_version")
    if not _is_int(sv):
        problems.append(f"{source}: schema_version must be an integer, got "
                        f"{sv!r}")
    elif sv != SNAPSHOT_SCHEMA_VERSION:
        problems.append(f"{source}: unsupported schema_version {sv!r}, "
                        f"expected {SNAPSHOT_SCHEMA_VERSION}")

    tf = facts.get("torch")
    if isinstance(tf, dict):
        for key in ("importable", "cuda_is_available"):
            if key in tf and not isinstance(tf[key], bool):
                problems.append(f"{source}: torch.{key} must be boolean, "
                                f"got {tf[key]!r}")
            elif key not in tf:
                problems.append(f"{source}: missing required field "
                                f"torch.{key}")
        if "device_count" not in tf:
            problems.append(f"{source}: missing required field "
                            "torch.device_count")
        elif not _is_int(tf["device_count"]) or tf["device_count"] < 0:
            problems.append(f"{source}: torch.device_count must be a "
                            f"non-negative integer, got {tf['device_count']!r}")
        # null is legitimate (version unknown / no HIP); anything else must be
        # a non-empty string, or analysis would "report" True/1/{} as a version
        for key in ("version", "hip_version"):
            if key in tf and tf[key] is not None:
                v = tf[key]
                if not isinstance(v, str) or not v.strip():
                    problems.append(f"{source}: torch.{key} must be null or a "
                                    f"non-empty string, got {v!r}")
    else:
        problems.append(f"{source}: 'torch' must be an object with "
                        "importable / version / hip_version / "
                        "cuda_is_available / device_count")

    devices = facts.get("devices")
    if devices is None:
        problems.append(f"{source}: missing required field 'devices' "
                        "(list of device fact objects)")
        devices = []
    elif not isinstance(devices, list):
        problems.append(f"{source}: 'devices' must be a list, got "
                        f"{type(devices).__name__}")
        devices = []
    seen: set = set()
    for i, d in enumerate(devices):
        where = f"{source}: devices[{i}]"
        if not isinstance(d, dict):
            problems.append(f"{where}: must be an object")
            continue
        idx = _need(d, "index", where, problems)
        if idx is not None:
            if not _is_int(idx) or idx < 0:
                problems.append(f"{where}: index must be a non-negative "
                                f"integer, got {idx!r}")
            elif idx in seen:
                problems.append(f"{where}: index {idx} duplicated")
            else:
                seen.add(idx)
        name = _need(d, "name", where, problems)
        if name is not None and not isinstance(name, str):
            problems.append(f"{where}: name must be a string or null, "
                            f"got {name!r}")
        gcn = _need(d, "gcn_arch_name", where, problems)
        if gcn is not None and not isinstance(gcn, str):
            problems.append(f"{where}: gcn_arch_name must be a string or "
                            f"null, got {gcn!r}")
        for key in ("total_memory", "shared_memory_per_block"):
            if key not in d:
                problems.append(f"{where}: missing required field '{key}' "
                                "(use null when the torch build does not "
                                "report it)")
            elif d[key] is not None and (not _is_int(d[key]) or d[key] < 0):
                problems.append(f"{where}: {key} must be a non-negative "
                                f"integer (bytes) or null, got {d[key]!r}")

    # 'rocm' is an optional object; when present it must not be a bare
    # string/list/bool or analysis would crash on .get / tuple comparison.
    if "rocm" in facts and facts["rocm"] is not None:
        rocm_f = facts["rocm"]
        if not isinstance(rocm_f, dict):
            problems.append(f"{source}: 'rocm' must be an object or null "
                            f"(it is optional), got {type(rocm_f).__name__}")
        else:
            for key in ("version", "version_file"):
                if key in rocm_f and rocm_f[key] is not None:
                    v = rocm_f[key]
                    if not isinstance(v, str) or not v.strip():
                        problems.append(
                            f"{source}: rocm.{key} must be null or a "
                            f"non-empty string, got {v!r}")
            if "parsed" in rocm_f and rocm_f["parsed"] is not None:
                pv = rocm_f["parsed"]
                if (not isinstance(pv, list) or len(pv) != 3
                        or not all(_is_int(x) and x >= 0 for x in pv)):
                    problems.append(
                        f"{source}: rocm.parsed must be null or exactly three "
                        f"non-negative integers [major, minor, patch], "
                        f"got {pv!r}")

    env = facts.get("env")
    if env is not None:
        if not isinstance(env, dict) or not all(
                isinstance(k, str) and isinstance(v, str)
                for k, v in env.items()):
            problems.append(f"{source}: 'env' must be an object mapping "
                            "variable names to string values")

    if problems:
        raise PreflightError(EXIT_USAGE, "invalid_snapshot", problems)
    return facts


def load_snapshot(path: str) -> dict:
    try:
        with open(path, encoding="utf8") as fp:
            raw = fp.read()
    except OSError as e:
        raise PreflightError(EXIT_USAGE, "invalid_snapshot",
                             [f"cannot read snapshot {path!r}: {e}"]) from e
    except UnicodeDecodeError as e:
        raise PreflightError(
            EXIT_USAGE, "invalid_snapshot",
            [f"snapshot {path!r} is not valid UTF-8 text (it must be a "
             f"UTF-8 JSON fact file): {e}"]) from e
    try:
        facts = json.loads(raw)
    except json.JSONDecodeError as e:
        raise PreflightError(EXIT_USAGE, "invalid_snapshot",
                             [f"snapshot {path!r} is not valid JSON: {e}"]) from e
    return validate_facts(facts, f"snapshot {path!r}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _diag(level: str, code: str, message: str) -> dict:
    return {"level": level, "code": code, "message": message}


def _fmt_bytes(n) -> str:
    if not isinstance(n, int):
        return "unknown"
    return f"{n} bytes ({n / 2**30:.1f} GiB)"


def analyze(facts: dict, device_index: int, env: dict, source: str) -> dict:
    """Pure analysis of collected facts -> report dict. No GPU/env mutation."""
    diagnostics: list[dict] = []
    report: dict = {
        "tool": TOOL_NAME,
        "report_version": REPORT_VERSION,
        "source": source,
        "requested_device": device_index,
        "status": None,
        "ok": False,
        "exit_code": EXIT_OK,
        "device": None,
        "software": {},
        "lds_expectation": None,
        "build_env": None,
        "binary_target_verification": {
            "state": "unknown",
            "reason": ("preflight never imports or inspects the compiled "
                       "exllamav3_ext, so the architecture it was actually "
                       "built for cannot be asserted here; environment "
                       "settings describe intent, not the binary"),
        },
        "recorded_env": {k: v for k, v in env.items() if k in RELEVANT_ENV_KEYS},
        "diagnostics": diagnostics,
        "next_steps": [],
    }

    tf = facts["torch"]
    report["software"] = {
        "torch_version": tf.get("version"),
        "hip_version": tf.get("hip_version"),
        "cuda_is_available": tf.get("cuda_is_available"),
        "device_count": tf.get("device_count"),
        "rocm_path_version": (facts.get("rocm") or {}).get("version"),
        "rocm_version_file": (facts.get("rocm") or {}).get("version_file"),
        "min_rocm_expected": ".".join(map(str, MIN_ROCM)),
    }

    # -- 1. HIP visibility ------------------------------------------------
    if not tf.get("importable"):
        report["status"] = "torch_not_importable"
        report["exit_code"] = EXIT_NO_HIP
        diagnostics.append(_diag(
            "error", "torch_not_importable",
            "torch could not be imported"
            + (f" ({tf.get('import_error')})" if tf.get("import_error") else "")
            + " -- no hardware facts are available. Install the ROCm build of "
              "torch (see requirements_rocm.txt) and rerun."))
        return _finish(report, facts, env, diagnostics)
    if tf.get("hip_version") is None:
        report["status"] = "not_rocm_torch"
        report["exit_code"] = EXIT_NO_HIP
        diagnostics.append(_diag(
            "error", "hip_missing",
            f"torch {tf.get('version')} reports no torch.version.hip -- this "
            "is not a ROCm build (or HIP runtime is missing). The EXL3 ROCm "
            "backend requires the ROCm build of torch "
            "(requirements_rocm.txt)."))
        return _finish(report, facts, env, diagnostics)
    devices = facts.get("devices") or []
    if not tf.get("cuda_is_available") or tf.get("device_count", 0) == 0:
        report["status"] = "no_hip_device_visible"
        report["exit_code"] = EXIT_NO_HIP
        diagnostics.append(_diag(
            "error", "hip_missing",
            "torch is a ROCm build but no HIP device is visible "
            f"(cuda.is_available={tf.get('cuda_is_available')}, "
            f"device_count={tf.get('device_count')}). Check the ROCm install "
            "(ROCm >= " + ".".join(map(str, MIN_ROCM)) + "), driver, and "
            "device nodes / container GPU passthrough."))
        if tf.get("query_error"):
            diagnostics.append(_diag("error", "query_error",
                                     str(tf["query_error"])))
        return _finish(report, facts, env, diagnostics)

    # -- ROCm version (setup.py gate) --------------------------------------
    rocm = facts.get("rocm") or {}
    parsed = rocm.get("parsed")
    if isinstance(parsed, list) and len(parsed) == 3:
        # Mirror setup.py::_check_rocm_version exactly: it tests raw truthiness
        # of EXL3_SKIP_ROCM_VERSION_CHECK (ANY non-empty string, even "0",
        # skips the gate) -- NOT the rocm_py _env_on semantics.
        if tuple(parsed) < MIN_ROCM and not env.get(
                "EXL3_SKIP_ROCM_VERSION_CHECK"):
            diagnostics.append(_diag(
                "warning", "rocm_below_minimum",
                f"ROCm {rocm.get('version')} at {rocm.get('version_file')} "
                f"is below the setup.py minimum {'.'.join(map(str, MIN_ROCM))}"
                " -- the build refuses to proceed without "
                "EXL3_SKIP_ROCM_VERSION_CHECK=1 (older ROCm computes wrong "
                "results with this port; override at your own risk)."))
    elif rocm.get("version") is None:
        where = (rocm.get("version_file")
                 if rocm.get("version_file") is not None
                 else "the snapshot has no 'rocm' block")
        diagnostics.append(_diag(
            "info", "rocm_version_unreadable",
            f"could not read a ROCm version from {where} -- setup.py warns "
            "and continues in the same situation."))

    # -- 2. device selection ------------------------------------------------
    valid = [d["index"] for d in devices if _is_int(d.get("index"))]
    if device_index not in valid:
        report["status"] = "invalid_device_selection"
        report["exit_code"] = EXIT_USAGE
        report["requested_device"] = device_index
        diagnostics.append(_diag(
            "error", "invalid_device_selection",
            f"--device {device_index} does not match any collected GPU. "
            f"Available device indices: {valid or 'none'}."))
        return _finish(report, facts, env, diagnostics)
    dev = devices[valid.index(device_index)]
    if dev.get("query_error") or not isinstance(dev.get("gcn_arch_name"), str):
        report["status"] = "invalid_device_selection"
        report["exit_code"] = EXIT_USAGE
        diagnostics.append(_diag(
            "error", "device_properties_unavailable",
            f"device {device_index} has no gcnArchName "
            + (f"({dev['query_error']})" if dev.get("query_error") else "")
            + " -- cannot normalize an architecture. Older torch builds "
              "expose gcnArchName only on ROCm."))
        return _finish(report, facts, env, diagnostics)

    arch, features = normalize_arch(dev["gcn_arch_name"])
    report["device"] = {
        "index": device_index,
        "name": dev.get("name"),
        "gcn_arch_name": dev["gcn_arch_name"],
        "arch": arch,
        "arch_features": features,
        "total_memory_bytes": dev.get("total_memory"),
        "shared_memory_per_block": dev.get("shared_memory_per_block"),
    }

    # -- 3. LDS expectation --------------------------------------------------
    expected = GPU_ARCH_SMEM.get(arch, DEFAULT_ARCH_SMEM)
    measured = dev.get("shared_memory_per_block")
    lds = {
        "arch": arch,
        "expected_shared_memory_per_block": expected,
        "measured_shared_memory_per_block": measured,
        "match": None,
        "expectation_source": "setup.py GPU_ARCH_SMEM (gfx1201 measured "
                              "64 KiB on real R9700 hardware)",
    }
    report["lds_expectation"] = lds
    if measured is None:
        diagnostics.append(_diag(
            "warning", "lds_unreported",
            "torch did not report shared_memory_per_block for this device, "
            f"so the {expected}-byte expectation could not be checked. "
            "Upgrade torch or query the device driver before building."))
    else:
        lds["match"] = measured == expected
        if arch == "gfx1201" and not lds["match"]:
            if measured < expected:
                lds["verdict"] = "inadequate"
                diagnostics.append(_diag(
                    "warning", "lds_inadequate",
                    f"gfx1201 reports shared_memory_per_block={measured} "
                    f"(expected 65536): kernels sized for 64 KiB of LDS will "
                    "fail shape admission. Investigate the driver / runtime "
                    "reporting before building or loading."))
            else:
                lds["verdict"] = "unexpected"
                diagnostics.append(_diag(
                    "warning", "lds_unexpected",
                    f"gfx1201 reports shared_memory_per_block={measured}, "
                    "not the 65536 measured on the reference R9700. A build "
                    "assuming 92160 for gfx1201 previously hit 'invalid "
                    "argument' setting the shared-memory attribute in "
                    "coop_autotune (doc/r9700_vs_v620.md issue 2); the fork "
                    "table now pins gfx1201 to 65536. Do not raise it again "
                    "without re-measuring."))
        elif not lds["match"]:
            diagnostics.append(_diag(
                "warning", "lds_mismatch",
                f"{arch} reports shared_memory_per_block={measured}, but "
                f"setup.py GPU_ARCH_SMEM expects {expected}. Verify the "
                "table before trusting either."))

    # -- 4. build-target environment vs device -------------------------------
    targets, tsource = resolve_build_targets(env)
    build_env = {
        "env_var_used": tsource,
        "raw_value": (env.get(tsource) if tsource else None),
        "target_archs": targets,
        "device_arch_in_targets": None,
        "note": ("environment describes the INTENDED build target only; it "
                 "is NOT evidence of the architecture of an already "
                 "compiled binary. Binary target verification stays "
                 "unknown from here."),
    }
    report["build_env"] = build_env
    if not targets:
        diagnostics.append(_diag(
            "info", "build_target_unset",
            "PYTORCH_ROCM_ARCH / GPU_ARCHS are unset: setup.py pins "
            "--offload-arch from rocminfo over SUPPORTED_GPU_ARCHS "
            "(hipcc auto-detect fallback). Set PYTORCH_ROCM_ARCH explicitly "
            f"to pin the target (this device is {arch})."))
    else:
        build_env["device_arch_in_targets"] = arch in targets
        if not build_env["device_arch_in_targets"]:
            diagnostics.append(_diag(
                "warning", "build_target_mismatch",
                f"{tsource} targets {targets}, which does not include the "
                f"selected device's arch {arch!r}. Building this way will "
                "not produce code for this GPU (setup.py's fat binary also "
                "takes the smallest LDS budget across targets, "
                "setup.py::_resolve_smem_max)."))
        else:
            diagnostics.append(_diag(
                "info", "build_target_matches_device",
                f"{tsource} targets include {arch!r}. Remember this is "
                "intent, not proof of the compiled binary's architecture."))

    # -- 5. fused-MoE environment (gfx120x) ----------------------------------
    is_gfx120x = arch in ("gfx1200", "gfx1201")
    forcing = env_on(env, "EXL3_ROCM_RDNA4_FUSED_MOE", False)
    moe = {"EXL3_ROCM_RDNA4_FUSED_MOE": {
        "value": env.get("EXL3_ROCM_RDNA4_FUSED_MOE"),
        "env_on": forcing,
        "applies_to_arch": is_gfx120x,
    }}
    report["fused_moe_env"] = moe
    if is_gfx120x:
        if forcing:
            report["status"] = "blocked_fused_moe_forcing"
            report["exit_code"] = EXIT_BLOCKED_ENV
            diagnostics.append(_diag(
                "error", "unsafe_fused_moe_forcing",
                f"EXL3_ROCM_RDNA4_FUSED_MOE={env.get('EXL3_ROCM_RDNA4_FUSED_MOE')!r} "
                "is ON under _env_on semantics on " + arch + ": it skips the "
                "rocm_py steer that keeps gfx120x MoE off the fused "
                "exl3_moe kernel, whose rdna_wmma path is a "
                "__builtin_trap() on gfx12 (no gfx11 WMMA encoding exists). "
                "Until a real gfx12 WMMA port lands this is a GPU-exception "
                "button, not a tuning knob. Unset it (or set it to 0) to "
                "take the default per-expert fallback path."))
        elif env.get("EXL3_ROCM_RDNA4_FUSED_MOE") is not None:
            diagnostics.append(_diag(
                "info", "fused_moe_steer_active",
                "EXL3_ROCM_RDNA4_FUSED_MOE is OFF under _env_on semantics: "
                "rocm_py keeps gfx120x MoE on the per-expert "
                "reconstruct+gemm path (correct, slower on MoE models)."))
    elif env.get("EXL3_ROCM_RDNA4_FUSED_MOE") is not None:
        diagnostics.append(_diag(
            "info", "fused_moe_flag_inert",
            f"EXL3_ROCM_RDNA4_FUSED_MOE is set but {arch} is not gfx120x; "
            "the rocm_py RDNA4 steer only consults it on gfx1200/gfx1201, "
            "so it has no effect on this device."))
    for name in ("EXL3_ROCM_MOE_MGEMM_ROUTE", "EXL3_ROCM_MOE_BSZN",
                 "EXL3_ROCM_MOE_DISABLE"):
        if name in env:
            moe[name] = {"value": env[name],
                         "env_on": env_on(env, name,
                                          name == "EXL3_ROCM_MOE_MGEMM_ROUTE")}
    if "EXL3_ROCM_MOE_MGEMM_MAX_ROWS" in env:
        # integer setting, not a boolean switch: value only, no _env_on reading
        moe["EXL3_ROCM_MOE_MGEMM_MAX_ROWS"] = {
            "value": env["EXL3_ROCM_MOE_MGEMM_MAX_ROWS"]}

    # -- 6. status ------------------------------------------------------------
    if report["status"] is None:
        if arch == "gfx1201":
            report["status"] = "comparison_workarounds_required"
            diagnostics.append(_diag(
                "info", "not_general_support",
                "R9700 facts collected. The library's gfx1201 support is NOT "
                "validated for general inference: the comparison results in "
                "doc/r9700_vs_v620.md required external adapters (MLP range "
                "balance, non-decode MoE route switch, alignment hint) that "
                "are not bundled here. Status is "
                "comparison_workarounds_required, not ready / "
                "general_supported. Exit 0 means the report was collected, "
                "not that inference works."))
        elif arch == "gfx1200":
            report["status"] = "rdna4_unvalidated"
            diagnostics.append(_diag(
                "warning", "rdna4_unvalidated",
                "gfx1200 is in setup.py SUPPORTED_GPU_ARCHS and the rocm_py "
                "gfx120x fused-MoE steer covers it, but this repo has no "
                "gfx1200 measurement record: the RDNA4 comparison in "
                "doc/r9700_vs_v620.md was run on gfx1201 only. Treat this "
                "part as unvalidated on real hardware."))
        elif arch in SUPPORTED_GPU_ARCHS:
            report["status"] = "not_r9700_supported_part"
            diagnostics.append(_diag(
                "info", "not_r9700",
                f"{arch} is in setup.py SUPPORTED_GPU_ARCHS but is not the "
                "R9700 (gfx1201) this preflight diagnoses. gfx1030-only "
                "tunings are gated per-arch in rocm_py; nothing here "
                "transfers tuning between cards."))
        else:
            report["status"] = "outside_supported_matrix"
            diagnostics.append(_diag(
                "warning", "outside_supported_matrix",
                f"{arch!r} is not in setup.py SUPPORTED_GPU_ARCHS "
                f"({sorted(SUPPORTED_GPU_ARCHS)}). setup.py refuses this "
                "target unless PYTORCH_ROCM_ARCH overrides it explicitly."))

    return _finish(report, facts, env, diagnostics)


def _rdna4_next_steps(arch: str, lds_match, measured) -> list[str]:
    """Build/steer advice for the SELECTED gfx120x architecture.

    gfx1201's 64 KiB figure is a measured fact; gfx1200's 92160 is a table
    assumption with no measurement record in this repo -- the two must not
    share advice text.
    """
    expected = GPU_ARCH_SMEM.get(arch, DEFAULT_ARCH_SMEM)
    steps = []
    if (arch == "gfx1201" and lds_match is False and measured is not None
            and measured < expected):
        steps.append(
            "Resolve the LDS report before building: gfx1201 must expose "
            "65536 shared_memory_per_block (measured on the reference "
            "R9700). An inadequate value here reproduces the "
            "coop_autotune shared-memory 'invalid argument' failure "
            "described in doc/r9700_vs_v620.md issue 2.")
    steps.append(
        f"Pin the build target explicitly: PYTORCH_ROCM_ARCH={arch} "
        "(GPU_ARCHS also works; PYTORCH_ROCM_ARCH wins). Unset, setup.py "
        "auto-detects via rocminfo over SUPPORTED_GPU_ARCHS. A multi-arch "
        "fat binary shares the SMALLEST per-arch LDS budget "
        "(setup.py::_resolve_smem_max).")
    if arch == "gfx1201":
        steps.append(
            "Keep the measured 64 KiB expectation for gfx1201: setup.py "
            "GPU_ARCH_SMEM pins it to 65536 (measured on the reference "
            "R9700) and rocm_tools/hipcc_probe.sh mirrors that value.")
    else:
        steps.append(
            f"Treat {arch}'s LDS number as a table-level assumption only: "
            f"setup.py GPU_ARCH_SMEM says {expected}, but this repo never "
            "measured the card -- re-measure shared_memory_per_block on the "
            "device first and do not reuse the R9700 64 KiB figure blindly.")
    steps.append(
        "rocm_tools/hipcc_probe.sh is an optional SOURCE-compilation "
        f"diagnostic (it compiles fresh exllamav3_ext translation units with "
        f"setup.py-style flags, e.g. GPU_ARCH={arch} rocm_tools/hipcc_probe.sh "
        "--all). Run it outside this preflight -- it writes build outputs and "
        "needs torch/hipcc, unlike this read-only tool. It does NOT inspect "
        "an already-installed extension binary, so the installed binary's "
        "architecture stays unknown here; verify that separately.")
    steps.append(
        "Leave EXL3_ROCM_RDNA4_FUSED_MOE unset (or 0). rocm_py steers "
        "gfx120x MoE to the per-expert path because the fused kernel's "
        "gfx11 WMMA intrinsics have no gfx12 encoding "
        "(rdna_wmma::mma_sync traps). Forcing it is for a future gfx12 "
        "WMMA port; this tool does not enable WMMA.")
    steps.append(
        "Read the caveats before judging performance: "
        "doc/r9700_vs_v620.md (external comparison adapters, alignment "
        "hint, MoE route workarounds -- none of them bundled) and "
        "doc/fork_changes.md 'What is not claimed here'. V620/gfx1030 "
        "tunings are arch-gated and are not applied here.")
    return steps


def _finish(report: dict, facts: dict, env: dict, diagnostics: list) -> dict:
    status = report["status"]
    dev = report.get("device") or {}
    arch = dev.get("arch")
    steps = []
    if status == "torch_not_importable":
        steps.append("Install the ROCm build of torch from "
                     "requirements_rocm.txt in the target environment, then "
                     "rerun this preflight (it only queries device "
                     "properties).")
    elif status in ("not_rocm_torch", "no_hip_device_visible"):
        steps.append("Fix HIP visibility first: ROCm >= "
                     + ".".join(map(str, MIN_ROCM)) + " (setup.py MIN_ROCM), "
                     "matching driver, and -- in containers -- GPU device "
                     "nodes/passthrough. requirements_rocm.txt pins the "
                     "ROCm torch build.")
    elif status == "invalid_device_selection":
        steps.append("Pick one of the available device indices shown above "
                     "with --device INDEX, or collect a snapshot that "
                     "includes the GPU you mean.")
    elif status == "blocked_fused_moe_forcing":
        steps.append("Unblock first: unset EXL3_ROCM_RDNA4_FUSED_MOE (or "
                     "set it to 0 / false / False -- _env_on treats any "
                     "other non-empty value as ON). The default rocm_py "
                     "per-expert MoE fallback is the only path that avoids "
                     "the gfx12 WMMA trap.")
        lds = report.get("lds_expectation") or {}
        steps.extend(_rdna4_next_steps(
            arch, lds.get("match"),
            lds.get("measured_shared_memory_per_block")))
    elif arch in ("gfx1201", "gfx1200") or status in (
            "comparison_workarounds_required", "rdna4_unvalidated"):
        lds = report.get("lds_expectation") or {}
        steps.extend(_rdna4_next_steps(
            arch, lds.get("match"),
            lds.get("measured_shared_memory_per_block")))
        if status == "rdna4_unvalidated":
            steps.append(f"{arch} has no measurement record in this repo "
                         "(only the R9700 is measured here). Budget for "
                         "first-run issues rather than assumed support.")
    elif status == "not_r9700_supported_part":
        steps.append("This part is supported by the port, but R9700-specific "
                     "diagnostics (64 KiB LDS expectation, gfx120x MoE "
                     "steer) do not apply to it. rocm_tools/README.md lists "
                     "per-card verification tools.")
    elif status == "outside_supported_matrix":
        steps.append("setup.py will refuse to build for this arch. Its "
                     "supported set is "
                     f"{sorted(SUPPORTED_GPU_ARCHS)}.")
    if facts.get("snapshot_env_missing_note"):
        steps.append("Add an 'env' block to snapshots if build/MoE "
                     "environment settings matter to the replay; without "
                     "one, all recorded env vars are treated as unset.")
    report["next_steps"] = steps
    report["ok"] = report["exit_code"] == EXIT_OK
    report["exit_note"] = ("exit 0 = the report was collected; it does NOT "
                           "prove EXL3 inference works on this device"
                           if report["exit_code"] == EXIT_OK else
                           "nonzero exit = diagnostic blocked this "
                           "configuration (see diagnostics)")
    return report


# ---------------------------------------------------------------------------
# Rendering / CLI
# ---------------------------------------------------------------------------

def render_text(report: dict) -> str:
    out = []
    out.append(f"{TOOL_NAME} -- read-only environment preflight "
               "(no build, no extension import, no model load)")
    out.append(f"source: {report['source']}")
    out.append(f"status: {report['status']}")
    sw = report.get("software") or {}
    out.append("")
    out.append("[software]")
    out.append(f"  torch: {sw.get('torch_version') or '-'}"
               f"   hip: {sw.get('hip_version') or '-'}")
    out.append(f"  ROCm version file: {sw.get('rocm_version_file') or '-'} "
               f"-> {sw.get('rocm_path_version') or 'unreadable'} "
               f"(expected >= {sw.get('min_rocm_expected')})")
    dev = report.get("device")
    if dev:
        out.append("")
        out.append(f"[device {dev['index']}]")
        out.append(f"  name: {dev['name']}")
        out.append(f"  gcnArchName: {dev['gcn_arch_name']} -> "
                   f"arch: {dev['arch']} (features: {dev['arch_features'] or 'none'})")
        out.append(f"  VRAM: {_fmt_bytes(dev['total_memory_bytes'])}")
        out.append(f"  shared_memory_per_block: "
                   f"{dev['shared_memory_per_block']}")
    lds = report.get("lds_expectation")
    if lds:
        verdict = ("MATCH" if lds.get("match") else
                   lds.get("verdict") or ("MISMATCH" if lds.get("match") is False
                                          else "unverified"))
        out.append(f"  LDS expectation ({lds['arch']}): expected "
                   f"{lds['expected_shared_memory_per_block']}, measured "
                   f"{lds['measured_shared_memory_per_block']} -> {verdict}")
    b = report.get("build_env")
    if b:
        out.append("")
        out.append("[build target environment]")
        if b["target_archs"]:
            out.append(f"  {b['env_var_used']} = {b['raw_value']!r} -> "
                       f"targets: {', '.join(b['target_archs'])}")
            out.append(f"  device arch in targets: "
                       f"{'yes' if b['device_arch_in_targets'] else 'no'}")
        else:
            out.append("  PYTORCH_ROCM_ARCH / GPU_ARCHS unset "
                       "(setup.py would auto-detect via rocminfo)")
        out.append(f"  note: {b['note']}")
    bt = report.get("binary_target_verification")
    if bt:
        out.append(f"  binary target verification: {bt.get('state')} "
                   f"({bt.get('reason')})")
    envrep = report.get("recorded_env") or {}
    out.append("")
    out.append("[recorded environment] (relevant keys only, no other env)")
    if envrep:
        for k in sorted(envrep):
            out.append(f"  {k}={envrep[k]!r}")
    else:
        out.append("  (no relevant environment variables set)")
    out.append("")
    out.append("[diagnostics]")
    for d in report["diagnostics"]:
        out.append(f"  {d['level'].upper()}: {d['message']}")
    if report.get("next_steps"):
        out.append("")
        out.append("[next steps]")
        for i, s in enumerate(report["next_steps"], 1):
            out.append(f"  {i}. {s}")
    out.append("")
    out.append("[not claimed by this tool]")
    out.append("  - does not enable RDNA4/gfx12 WMMA (the fused-MoE WMMA "
               "path traps on gfx1201 today)")
    out.append("  - does not apply V620/gfx1030-only tunings to other cards")
    out.append("  - does not load a model, run a kernel, or measure "
               "throughput: this is startup diagnosis, not a benchmark")
    out.append(f"  {report.get('exit_note')}")
    return "\n".join(out)


def main(argv=None, out=None, err=None) -> int:
    parser = argparse.ArgumentParser(
        prog="r9700_preflight.py",
        description="Read-only R9700 (gfx1201) environment preflight: "
                    "torch/HIP versions, device arch and LDS facts, build "
                    "target and fused-MoE environment checks. Reports a "
                    "diagnosis; it does not build, load, or benchmark "
                    "anything.",
        epilog="exit codes: 0 report collected (NOT inference proven), "
               "1 invalid snapshot/device selection, 2 HIP/torch missing, "
               "3 blocked unsafe fused-MoE forcing. Examples: "
               "python3 rocm_tools/r9700_preflight.py --json; "
               "python3 rocm_tools/r9700_preflight.py --snapshot snap.json "
               "--device 0; see doc/r9700_preflight.md for the snapshot "
               "schema.",
    )
    parser.add_argument("--device", type=int, default=0, metavar="INDEX",
                        help="CUDA/HIP device index to diagnose (default: 0)")
    parser.add_argument("--json", action="store_true",
                        help="print the report as machine-readable JSON")
    parser.add_argument("--snapshot", metavar="FILE",
                        help="read collected facts from an offline JSON "
                             "snapshot instead of querying torch (schema: "
                             "doc/r9700_preflight.md)")
    args = parser.parse_args(argv)
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr

    try:
        if args.snapshot:
            facts = load_snapshot(args.snapshot)
            env = facts.get("env")
            if env is None:
                env = {}
                facts["snapshot_env_missing_note"] = True
            source = f"snapshot {args.snapshot!r}"
        else:
            env = dict(os.environ)
            facts = collect_live_facts(env)
            validate_facts(facts, "live collection")
            source = "live torch query"
        report = analyze(facts, args.device, env, source)
    except PreflightError as e:
        report = {
            "tool": TOOL_NAME, "report_version": REPORT_VERSION,
            "source": args.snapshot or "live torch query",
            "requested_device": args.device, "status": e.status,
            "ok": False, "exit_code": e.exit_code,
            "diagnostics": [_diag("error", e.status, m) for m in e.messages],
            "next_steps": [],
        }
        if args.json:
            out.write(json.dumps(report, indent=2) + "\n")
        else:
            for d in report["diagnostics"]:
                err.write(f"{TOOL_NAME}: {d['level']}: {d['message']}\n")
        return report["exit_code"]

    if args.json:
        out.write(json.dumps(report, indent=2) + "\n")
    else:
        out.write(render_text(report) + "\n")
        for d in report["diagnostics"]:
            if d["level"] == "error":
                err.write(f"{TOOL_NAME}: error report collected "
                          f"(exit {report['exit_code']}, "
                          f"status {report['status']})\n")
                break
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
